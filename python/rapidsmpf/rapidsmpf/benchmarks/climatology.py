# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compute a climatology.

These are average values (e.g. for temperature) at each spatial location, grouped
by some time period (e.g. day of year, hour of day).

The raw data are stored chunked by time, and contiguous or chunked in space depending
on the resolution of the dataset: `{time=1, lat=721, lon=1440, level=13}`.

To compute a climatology, we need to gather all the data for a particular area
across time. i.e. we need to rechunk such to something like `{time=-1, lat=auto, lon=auto, level=13}`,
where lat / lon are rechunked to something that keeps memory usage roughly constant.

Consider the tasks assigned to an individual node:

1. Load some *chunk* (ndarray) of data from the input Zarr store into GPU memory.
2. *Collectively* shuffle the chunks to their assigned nodes
  - Just slice the subset of our in-memory chunks for our spatial region
  - *Send* the other spatial regions to the other nodes
  - *Receive* our spatial region from the other nodes
3. Reduce the data (over time) for the spatial region we've been assigned
  - Each spatial reduction is embarrassingly parallel
4. Rechunk back to `{dayofyear=1, hour=1, lat=721, lon=1440, level=13}`

See https://github.com/coiled/benchmarks/discussions/1545#discussioncomment-10619682 for more.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import zarr

import rmm

import rapidsmpf.communicator.single
from rapidsmpf.benchmarks.geo import generate_data, parse_shape
from rapidsmpf.config import Options
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.chunks.arbitrary import ArbitraryChunk
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import (
    define_py_node,
    run_streaming_pipeline,
)

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.node import CppNode, PyNode


@dataclass
class ArrayChunk:
    """
    A chunk of array data with metadata.

    This is a simple wrapper around a cupy array with shape/dtype metadata
    for passing through channels without any serialization overhead.
    """

    data: cp.ndarray  # The actual data (cupy array on GPU)
    partition_id: int  # The target partition ID for this chunk
    time_index: int  # The time index this data came from


@dataclass
class ReducedChunk:
    """A reduced chunk ready to be written."""

    partition_id: int
    lat_slice: slice
    lon_slice: slice
    data: np.ndarray  # Reduced data (numpy array, back on host)


def assign_read_tasks(
    array: zarr.Array, rank: int, n_ranks: int
) -> list[tuple[int, ...]]:
    """
    Assign read tasks to nodes.

    Parameters
    ----------
    array
        The array to assign tasks to.
    rank
        The rank of the current node.
    n_ranks
        The total number of ranks.

    Returns
    -------
    list[tuple[int, ...]]
        A list of tuples, indicating the block (chunk index) to read for each dimension.
    """
    # Just a round-robin assignment of tasks to nodes.
    n_chunks = [
        math.ceil(s / c) for s, c in zip(array.shape, array.chunks, strict=True)
    ]
    block_selections = list(itertools.product(*[range(n) for n in n_chunks]))

    return [block for i, block in enumerate(block_selections) if i % n_ranks == rank]


def compute_spatial_partitions(
    lat_size: int, lon_size: int, num_partitions: int
) -> list[tuple[slice, slice]]:
    """
    Compute spatial partition slices.

    Divides the lat/lon grid into roughly equal partitions.

    Parameters
    ----------
    lat_size
        Size of the latitude dimension.
    lon_size
        Size of the longitude dimension.
    num_partitions
        Number of partitions to create.

    Returns
    -------
    list[tuple[slice, slice]]
        List of (lat_slice, lon_slice) for each partition.
    """
    # Simple 1D partitioning along latitude for now
    lat_per_part = math.ceil(lat_size / num_partitions)

    partitions = []
    for i in range(num_partitions):
        lat_start = i * lat_per_part
        lat_end = min((i + 1) * lat_per_part, lat_size)
        if lat_start < lat_size:
            partitions.append((slice(lat_start, lat_end), slice(None)))

    return partitions


@define_py_node()
async def read_chunks_node(
    ctx: Context,
    ch_out: Channel[ArbitraryChunk],
    array: zarr.Array,
    blocks: list[tuple[int, ...]],
    num_producers: int,
    partition_slices: list[tuple[slice, slice]],
) -> None:
    """
    Read chunks from the input Zarr store with throttling.

    Uses asyncio.Semaphore to limit concurrent read operations.
    Partitions each chunk spatially and sends one message per partition.

    Parameters
    ----------
    ctx
        The streaming context.
    ch_out
        Output channel for read chunks.
    array
        The Zarr array to read from.
    blocks
        List of block indices to read.
    num_producers
        Maximum number of concurrent read operations.
    partition_slices
        List of (lat_slice, lon_slice) for spatial partitioning.
    """
    throttle = asyncio.Semaphore(num_producers)
    seq_num = 0

    async def read_and_send(block: tuple[int, ...]) -> None:
        nonlocal seq_num
        async with throttle:
            # Read from Zarr (this is synchronous I/O)
            host_data: np.ndarray = np.asarray(array.get_block_selection(block))

            # Transfer to GPU
            gpu_data = cp.asarray(host_data)

            time_index = block[0]  # Assuming time is first dimension

            # Partition spatially and send one message per partition
            for pid, (lat_slice, lon_slice) in enumerate(partition_slices):
                # Slice spatial dimensions (assuming shape is [time, lat, lon, ...])
                if gpu_data.ndim >= 3:
                    sliced = gpu_data[:, lat_slice, lon_slice, ...]
                elif gpu_data.ndim == 2:
                    sliced = gpu_data[lat_slice, lon_slice]
                else:
                    sliced = gpu_data

                if sliced.size > 0:
                    # Create chunk with cupy array directly - no serialization
                    chunk = ArrayChunk(
                        data=sliced,
                        partition_id=pid,
                        time_index=time_index,
                    )
                    msg = Message(seq_num, ArbitraryChunk(chunk))
                    seq_num += 1
                    await ch_out.send(ctx, msg)

    # Process all blocks
    tasks = [read_and_send(block) for block in blocks]
    await asyncio.gather(*tasks)

    await ch_out.drain(ctx)


@define_py_node()
async def reduce_node(
    ctx: Context,
    ch_in: Channel[ArbitraryChunk],
    ch_out: Channel[ArbitraryChunk],
    partition_slices: list[tuple[slice, slice]],
    num_partitions: int,
) -> None:
    """
    Incrementally reduce chunks over the time dimension.

    Accumulates sum and count for each partition, computes mean at the end.
    This avoids materializing all data at once to prevent OOM.

    Parameters
    ----------
    ctx
        The streaming context.
    ch_in
        Input channel with array chunks.
    ch_out
        Output channel for reduced chunks.
    partition_slices
        List of (lat_slice, lon_slice) for each partition.
    num_partitions
        Number of spatial partitions.
    """
    # Incremental reduction state: sum and count per partition
    # We store (running_sum, count) for each partition
    partition_accum: dict[int, tuple[cp.ndarray | None, int]] = dict.fromkeys(
        range(num_partitions), (None, 0)
    )

    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = ArbitraryChunk.from_message(msg)
        array_chunk: ArrayChunk = chunk.release()

        pid = array_chunk.partition_id
        data = array_chunk.data

        running_sum, count = partition_accum[pid]

        if running_sum is None:
            # First chunk for this partition - initialize
            running_sum = data.astype(cp.float64)
        else:
            # Accumulate
            running_sum += data.astype(cp.float64)

        partition_accum[pid] = (running_sum, count + 1)

    # Compute mean and send reduced chunks
    for pid in range(num_partitions):
        running_sum, count = partition_accum[pid]

        if running_sum is not None and count > 0:
            # Compute mean over the time dimension
            # running_sum has shape [1, lat_slice_size, lon_slice_size, ...]
            # We reduce over the first axis and divide by count
            reduced = (running_sum / count).astype(cp.float32)

            # Squeeze out the time dimension if present
            if reduced.ndim > 0 and reduced.shape[0] == 1:
                reduced = reduced.squeeze(axis=0)

            lat_slice, lon_slice = partition_slices[pid]

            # Move back to host for writing
            reduced_host = cp.asnumpy(reduced)

            reduced_chunk = ReducedChunk(
                partition_id=pid,
                lat_slice=lat_slice,
                lon_slice=lon_slice,
                data=reduced_host,
            )

            msg = Message(pid, ArbitraryChunk(reduced_chunk))
            await ch_out.send(ctx, msg)

    await ch_out.drain(ctx)


@define_py_node()
async def write_chunks_node(
    ctx: Context,
    ch_in: Channel[ArbitraryChunk],
    output_array: zarr.Array,
) -> None:
    """
    Write reduced chunks to the output Zarr store.

    Parameters
    ----------
    ctx
        The streaming context.
    ch_in
        Input channel with reduced chunks.
    output_array
        The Zarr array to write to.
    """
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = ArbitraryChunk.from_message(msg)
        reduced_chunk: ReducedChunk = chunk.release()

        lat_slice = reduced_chunk.lat_slice
        lon_slice = reduced_chunk.lon_slice
        data = reduced_chunk.data

        # Write to the appropriate region of the output array
        output_array[lat_slice, lon_slice, ...] = data


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    args
        Optional list of arguments to parse. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="climatology",
        description="Benchmark for climatology operations",
    )

    # {'time': 93544, 'latitude': 721, 'longitude': 1440, 'level': 13}
    parser.add_argument(
        "-s",
        "--shape",
        type=parse_shape,
        default=(365, 1440, 721, 13),
        help=(
            "Input array shape as 'time,longitude,latitude'. Default: '365,1440,721,13'"
        ),
    )

    parser.add_argument(
        "-i",
        "--input-path",
        type=Path,
        default=Path("data.zarr"),
        help="Path to the root of the input zarr store. Default: 'data.zarr'",
    )

    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help=("Path to the root of the output zarr store. Default: same as input-path"),
    )

    parser.add_argument(
        "--input-group",
        type=str,
        default="climatology-in",
        help="Group name inside the zarr store for input data. Default: 'climatology-in'",
    )

    parser.add_argument(
        "--output-group",
        type=str,
        default="climatology-out",
        help="Group name inside the zarr store for output data. Default: 'climatology-out'",
    )

    parser.add_argument(
        "-n",
        "--num-producers",
        type=int,
        default=4,
        help="Number of concurrent read operations (throttling). Default: 4",
    )

    parser.add_argument(
        "-p",
        "--num-partitions",
        type=int,
        default=4,
        help="Number of spatial partitions for shuffling. Default: 4",
    )

    parsed = parser.parse_args(args)

    # Default output-path to input-path if not specified
    if parsed.output_path is None:
        parsed.output_path = parsed.input_path

    return parsed


def main(args: list[str] | None = None) -> None:
    """
    Entry point for the climatology benchmark.

    Parameters
    ----------
    args
        Optional list of command line arguments.
    """
    parsed = parse_args(args)

    # Generate input data if needed
    generate_data(
        input_path=parsed.input_path,
        input_group=parsed.input_group,
        shape=parsed.shape,
    )

    # Open input data
    root = zarr.open_group(parsed.input_path, mode="a")
    input_group: zarr.Group = root[parsed.input_group]  # type: ignore[assignment]
    input_array: zarr.Array = input_group["temperature"]  # type: ignore[assignment]

    # Create output group and array
    if parsed.output_group not in root:
        output_group = root.create_group(parsed.output_group)
    else:
        output_group = root[parsed.output_group]

    assert isinstance(output_group, zarr.Group)

    # Output shape: same spatial dims, no time dimension (reduced)
    # Assuming input shape is (time, lat, lon, ...)
    output_shape = input_array.shape[1:]  # Remove time dimension
    if "temperature_climatology" not in output_group:
        output_array = output_group.create_array(
            "temperature_climatology",
            shape=output_shape,
            chunks=output_shape,  # Single chunk for simplicity
            dtype=np.float32,
        )
    else:
        output_array = output_group["temperature_climatology"]

    assert isinstance(output_array, zarr.Array)

    # Initialize rapidsmpf
    options = Options()
    statistics = Statistics(enable=True)

    br = BufferResource(rmm.mr.CudaMemoryResource())
    comm = rapidsmpf.communicator.single.new_communicator(options)
    ctx = Context(
        comm=comm,
        br=br,
        options=options,
        statistics=statistics,
    )

    # Assign read tasks to this rank
    rank = ctx.comm().rank
    n_ranks = ctx.comm().nranks
    blocks = assign_read_tasks(input_array, rank, n_ranks)

    # Compute spatial partitions
    lat_size = input_array.shape[1]  # Assuming shape is (time, lat, lon, ...)
    lon_size = input_array.shape[2]
    partition_slices = compute_spatial_partitions(
        lat_size, lon_size, parsed.num_partitions
    )

    # Create channels
    ch_read: Channel[ArbitraryChunk] = ctx.create_channel()
    ch_reduce: Channel[ArbitraryChunk] = ctx.create_channel()

    # Create nodes - simplified pipeline: read -> reduce -> write
    nodes: list[CppNode | PyNode] = []

    # Read node with throttling and spatial partitioning
    nodes.append(
        read_chunks_node(
            ctx,
            ch_out=ch_read,
            array=input_array,
            blocks=blocks,
            num_producers=parsed.num_producers,
            partition_slices=partition_slices,
        )
    )

    # Reduce node - incremental reduction over time
    nodes.append(
        reduce_node(
            ctx,
            ch_in=ch_read,
            ch_out=ch_reduce,
            partition_slices=partition_slices,
            num_partitions=parsed.num_partitions,
        )
    )

    # Write node
    nodes.append(
        write_chunks_node(
            ctx,
            ch_in=ch_reduce,
            output_array=output_array,
        )
    )

    # Create executor for Python nodes
    py_executor = ThreadPoolExecutor(max_workers=1)

    # Run the streaming pipeline
    print(f"Running climatology pipeline with {len(blocks)} blocks on rank {rank}")
    print(f"  Throttling: {parsed.num_producers} concurrent reads")
    print(f"  Partitions: {parsed.num_partitions}")

    run_streaming_pipeline(nodes=nodes, py_executor=py_executor)

    # Print statistics
    print("\nPipeline completed!")
    print(f"Output written to {parsed.output_path}/{parsed.output_group}")

    py_executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
