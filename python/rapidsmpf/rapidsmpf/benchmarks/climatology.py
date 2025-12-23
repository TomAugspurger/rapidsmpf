# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import itertools
import math
from pathlib import Path
from typing import TYPE_CHECKING

import zarr

import rmm

import rapidsmpf.communicator.single
from rapidsmpf.benchmarks.geo import generate_data, parse_shape
from rapidsmpf.config import Options
from rapidsmpf.memory.buffer_resource import BufferResource
from rapidsmpf.statistics import Statistics
from rapidsmpf.streaming.core.context import Context
from rapidsmpf.streaming.core.node import define_py_node

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.message import Message, PayloadT


class ZarrPayload:
    """
    Payload for zarr.
    """

    @classmethod
    def from_message(cls: type[PayloadT], message: Message[PayloadT]) -> PayloadT: ...
    def into_message(
        self: PayloadT, sequence_number: int, message: Message[PayloadT]
    ) -> None: ...


def assign_read_tasks(
    array: zarr.Array, rank: int, n_ranks: int
) -> list[tuple[int, ...]]:
    """
    Assign tasks to nodes.

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
    list[tuple[int, int, int]]
        A list of tuples, indicating the block (chunk index) to read for each dimension.
    """
    # Just a round-robin assignment of tasks to nodes.
    n_chunks = [
        math.ceil(s / c) for s, c in zip(array.shape, array.chunks, strict=True)
    ]
    block_selections = list(itertools.product(*[range(n) for n in n_chunks]))

    return [block for i, block in enumerate(block_selections) if i % n_ranks == rank]


def assign_reduce_tasks(
    array: zarr.Array, rank: int, n_ranks: int
) -> list[tuple[int, int, int]]:
    """
    Assign reduce tasks to nodes.

    The reduce tasks are on the rechunked array.
    """
    return []


@define_py_node()
async def read_chunks(
    ctx: Context,
    ch_out: Channel,
    ch_in: Channel,
    array: zarr.Array,
    blocks: list[tuple[int, int, int]],
) -> None:
    """
    Read chunks from the input Zarr store.
    """
    # read to host memory
    # TODO: pinned host memory
    # TODO: stream
    # TODO: Semaphore
    # something is wrong with zarr's types here
    host_chunks = [array.get_block_selection([block]) for block in blocks]  # type: ignore[arg-type]

    # TODO: DtoH copy

    # TODO: send messages


@define_py_node()
async def gather_chunks(
    ctx: Context, ch_out: Channel, ch_in: Channel, blocks: list[tuple[int, int, int]]
) -> None:
    """
    Gather the chunks for a reduction.
    """
    # TODO: receive messages
    # TODO: reduce


@define_py_node()
async def reduce_chunks(
    ctx: Context, ch_out: Channel, ch_in: Channel, blocks: list[tuple[int, int, int]]
) -> None: ...


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
        help="Group name inside the zarr store for input data. Default: 'input'",
    )

    parser.add_argument(
        "--output-group",
        type=str,
        default="climatology-out",
        help="Group name inside the zarr store for output data. Default: 'output'",
    )

    parsed = parser.parse_args(args)

    # Default output-path to input-path if not specified
    if parsed.output_path is None:
        parsed.output_path = parsed.input_path

    return parsed


def main(args: list[str] | None = None) -> None:
    """
    Entry point for the geo-regridding benchmark.

    Parameters
    ----------
    args
        Optional list of command line arguments.
    """
    parsed = parse_args(args)
    generate_data(
        input_path=parsed.input_path,
        input_group=parsed.input_group,
        shape=parsed.shape,
    )
    root = zarr.open_group(parsed.input_path, mode="r")

    # initialize rapidsmpf things
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
    group: zarr.Group = root["climatology-in"]  # type: ignore[assignment]
    array: zarr.Array = group["temperature"]  # type: ignore[assignment]

    blocks = assign_read_tasks(array, ctx.comm().rank, 2)

    # Setup the channels...
    ctx.create_channel()


if __name__ == "__main__":
    main()
