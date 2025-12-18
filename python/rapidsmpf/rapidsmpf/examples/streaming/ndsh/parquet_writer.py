# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Parquet writer operators for NDS-H streaming queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc

from rapidsmpf.examples.streaming.ndsh.utils import to_device
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk
from rapidsmpf.utils.cudf import pylibcudf_to_cudf_dataframe

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.message import Message


@define_py_node()
async def write_parquet(
    ctx: Context,
    ch_in: Channel[TableChunk],
    output_path: str,
    column_names: list[str],
) -> None:
    """
    Write chunks in a channel to an output parquet file.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Input channel of TableChunks.
    output_path
        Path to write the parquet file.
    column_names
        Names of the columns to add to the parquet metadata.
    """
    # Collect all chunks first, then write
    tables: list[plc.Table] = []
    chunks: list[TableChunk] = []

    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        tables.append(chunk.table_view())
        chunks.append(chunk)

    if len(tables) == 0:
        # Write empty file
        empty_table = plc.Table([])
        df = pylibcudf_to_cudf_dataframe(empty_table, column_names=[])
        df.to_parquet(output_path)
    elif len(tables) == 1:
        # Single table
        df = pylibcudf_to_cudf_dataframe(tables[0], column_names=column_names)
        df.to_parquet(output_path)
    else:
        # Concatenate and write
        result_table = plc.concatenate.concatenate(tables)
        df = pylibcudf_to_cudf_dataframe(result_table, column_names=column_names)
        df.to_parquet(output_path)
