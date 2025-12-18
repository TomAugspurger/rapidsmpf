# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for NDS-H streaming queries."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rapidsmpf.streaming.core.node import define_py_node

if TYPE_CHECKING:
    from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk


def list_parquet_files(root_path: str | Path) -> list[str]:
    """
    List all parquet files in a given path.

    Parameters
    ----------
    root_path
        The path to look in.

    Returns
    -------
    list[str]
        If `root_path` names a regular file that ends with `.parquet` then a singleton
        list of just that file. If `root_path` is a directory, then a list containing
        all regular files in that directory whose name ends with `.parquet`.

    Raises
    ------
    RuntimeError
        If the `root_path` doesn't name a regular file or a directory,
        or if it names a regular file that doesn't end in `.parquet`.
    """
    root = Path(root_path)

    if not root.exists():
        raise RuntimeError(f"Invalid file path: {root_path}")

    if root.is_file():
        if not str(root).endswith(".parquet"):
            raise RuntimeError(f"Invalid filename: {root_path}")
        return [str(root)]

    if root.is_dir():
        return [
            str(entry)
            for entry in root.iterdir()
            if entry.is_file() and entry.name.endswith(".parquet")
        ]

    raise RuntimeError(f"Invalid file path: {root_path}")


def get_table_path(input_directory: str, table_name: str) -> str:
    """
    Get the path to a given table.

    Parameters
    ----------
    input_directory
        Input directory.
    table_name
        Name of table to find.

    Returns
    -------
    str
        Path to given table.
    """
    directory = input_directory if input_directory else "."
    file_path = Path(f"{directory}/{table_name}.parquet")

    if file_path.exists():
        return str(file_path)

    return f"{directory}/{table_name}/"


def to_device(
    ctx: Context,
    chunk: TableChunk,
    *,
    allow_overbooking: bool = False,
) -> TableChunk:
    """
    Ensure a TableChunk is on device.

    Parameters
    ----------
    ctx
        Streaming context.
    chunk
        Chunk to move to device; is left in a moved-from state.
    allow_overbooking
        Whether reserving memory is allowed to overbook.

    Returns
    -------
    TableChunk
        New TableChunk on device.

    Raises
    ------
    OverflowError
        If overbooking is not allowed and not enough memory is available to reserve.
    """
    return chunk.make_available_and_spill(ctx.br(), allow_overbooking=allow_overbooking)


@define_py_node()
async def sink_channel(
    ctx: Context,
    ch: Channel[TableChunk],
) -> None:
    """
    Sink messages into a channel and discard them.

    Parameters
    ----------
    ctx
        Streaming context.
    ch
        Channel to discard messages from.
    """
    await ch.shutdown(ctx)


@define_py_node()
async def consume_channel(
    ctx: Context,
    ch_in: Channel[TableChunk],
) -> None:
    """
    Consume messages from a channel and discard them.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Channel to consume messages from.

    Notes
    -----
    If the channel contains TableChunks, moves them to device and prints
    small amount of detail about them (row and column count).
    """
    from rapidsmpf.streaming.cudf.table_chunk import TableChunk

    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        ctx.comm().logger.print(
            f"Consumed chunk with {chunk.table_view().num_rows()} rows "
            f"and {chunk.table_view().num_columns()} columns"
        )


class CommType(Enum):
    """Communicator type to use."""

    SINGLE = "single"
    MPI = "mpi"
    UCXX = "ucxx"


@dataclass
class ProgramOptions:
    """Configuration options for the query."""

    num_streaming_threads: int = 1
    num_iterations: int = 2
    num_streams: int = 16
    comm_type: CommType = CommType.UCXX
    periodic_spill_ms: int | None = None
    num_rows_per_chunk: int = 100_000_000
    spill_device_limit: float | None = None
    no_pinned_host_memory: bool = False
    use_shuffle_join: bool = False
    output_file: str = ""
    input_directory: str = ""


def parse_arguments(args: list[str] | None = None) -> ProgramOptions:
    """
    Parse commandline arguments.

    Parameters
    ----------
    args
        Arguments to parse. If None, uses sys.argv.

    Returns
    -------
    ProgramOptions
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NDS-H Query Benchmark")

    parser.add_argument(
        "--num-streaming-threads",
        type=int,
        default=1,
        help="Number of streaming threads (default: 1)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=2,
        help="Number of iterations (default: 2)",
    )
    parser.add_argument(
        "--num-streams",
        type=int,
        default=16,
        help="Number of streams in stream pool (default: 16)",
    )
    parser.add_argument(
        "--num-rows-per-chunk",
        type=int,
        default=100_000_000,
        help="Number of rows per chunk (default: 100,000,000)",
    )
    parser.add_argument(
        "--spill-device-limit",
        type=float,
        default=None,
        help="Fractional spill device limit (0.0-1.0)",
    )
    parser.add_argument(
        "--no-pinned-host-memory",
        action="store_true",
        help="Disable pinned host memory",
    )
    parser.add_argument(
        "--periodic-spill",
        type=int,
        default=None,
        help="Duration in milliseconds between periodic spilling checks",
    )
    parser.add_argument(
        "--comm-type",
        type=str,
        default="ucxx",
        choices=["single", "mpi", "ucxx"],
        help="Communicator type (default: ucxx)",
    )
    parser.add_argument(
        "--use-shuffle-join",
        action="store_true",
        help="Use shuffle join for 'big' joins",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--input-directory",
        type=str,
        required=True,
        help="Input directory path",
    )

    parsed = parser.parse_args(args)

    return ProgramOptions(
        num_streaming_threads=parsed.num_streaming_threads,
        num_iterations=parsed.num_iterations,
        num_streams=parsed.num_streams,
        comm_type=CommType(parsed.comm_type),
        periodic_spill_ms=parsed.periodic_spill,
        num_rows_per_chunk=parsed.num_rows_per_chunk,
        spill_device_limit=parsed.spill_device_limit,
        no_pinned_host_memory=parsed.no_pinned_host_memory,
        use_shuffle_join=parsed.use_shuffle_join,
        output_file=parsed.output_file,
        input_directory=parsed.input_directory,
    )


def create_context(
    options: ProgramOptions,
    mr: RmmResourceAdaptor,
) -> Context:
    """
    Create a streaming execution context for a query.

    Parameters
    ----------
    options
        Arguments to configure the context.
    mr
        Memory resource to use for all allocations.

    Returns
    -------
    Context
        Shared streaming context.

    Warning
    -------
    The memory resource must be kept alive until the final usage of the
    returned Context is complete.
    """
    from mpi4py import MPI

    import rmm.mr

    from rapidsmpf.communicator.mpi import new_communicator as mpi_communicator
    from rapidsmpf.communicator.single import new_communicator as single_communicator
    from rapidsmpf.config import Options, get_environment_variables
    from rapidsmpf.memory.buffer import MemoryType
    from rapidsmpf.memory.buffer_resource import BufferResource, LimitAvailableMemory
    from rapidsmpf.statistics import Statistics
    from rapidsmpf.streaming.core.context import Context

    rmm.mr.set_current_device_resource(mr)  # type: ignore[arg-type]

    # Build memory_available limits
    memory_available: dict[MemoryType, LimitAvailableMemory] = {}
    if options.spill_device_limit is not None:
        total_mem = rmm.mr.available_device_memory()[1]
        limit_size = int(total_mem * options.spill_device_limit)
        memory_available[MemoryType.DEVICE] = LimitAvailableMemory(mr, limit=limit_size)

    statistics = Statistics(enable=True, mr=mr)

    br = BufferResource(
        mr,  # type: ignore[arg-type]
        memory_available=memory_available if memory_available else None,
    )

    # Build config options
    env_vars = get_environment_variables()
    env_vars["NUM_STREAMING_THREADS"] = str(options.num_streaming_threads)
    config_options = Options(env_vars)

    # Create communicator
    if options.comm_type == CommType.SINGLE:
        comm = single_communicator(config_options)
    elif options.comm_type == CommType.MPI:
        comm = mpi_communicator(MPI.COMM_WORLD, config_options)
    elif options.comm_type == CommType.UCXX:
        # For UCXX, use MPI-based initialization
        raise NotImplementedError("UCXX communicator is not supported")
        # from rapidsmpf.communicator import ucxx

        # comm = ucxx.init_using_mpi(MPI.COMM_WORLD, config_options)
    else:
        raise ValueError(f"Unknown communicator type: {options.comm_type}")

    ctx = Context(
        comm=comm,
        br=br,
        options=config_options,
        statistics=statistics,
    )

    if comm.rank == 0:
        comm.logger.print(f"Execution context on {comm.nranks} ranks")

    return ctx
