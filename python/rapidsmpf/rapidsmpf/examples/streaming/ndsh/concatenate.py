# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Concatenation operators for NDS-H streaming queries."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pylibcudf as plc

from rapidsmpf.examples.streaming.ndsh.utils import to_device
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


class ConcatOrder(Enum):
    """Specify whether the concatenation should respect input ordering."""

    DONT_CARE = "dont_care"
    LINEARIZE = "linearize"


@define_py_node()
async def concatenate(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    order: ConcatOrder = ConcatOrder.DONT_CARE,
) -> None:
    """
    Concatenate all table chunks from an input channel.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Input channel of TableChunks.
    ch_out
        Output channel of concatenated chunks, contains at most one message.
    order
        Do we care about maintaining the input ordering?
    """
    messages: list[Message[TableChunk]] = []
    concat_stream = ctx.get_stream_from_pool()

    # Collect all messages
    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        messages.append(msg)

    if len(messages) == 0:
        # Send empty table
        empty_table = plc.Table([])
        result_chunk = TableChunk.from_pylibcudf_table(
            empty_table, concat_stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(0, result_chunk))
    elif len(messages) == 1:
        # Just forward the single message
        await ch_out.send(ctx, messages[0])
    else:
        # Sort by sequence number if linearizing
        if order == ConcatOrder.LINEARIZE:
            messages.sort(key=lambda m: m.sequence_number)

        # Collect chunks and views
        chunks: list[TableChunk] = []
        views: list[plc.Table] = []
        for msg in messages:
            chunk = TableChunk.from_message(msg)
            chunk = to_device(ctx, chunk)
            views.append(chunk.table_view())
            chunks.append(chunk)

        # Concatenate all tables
        result_table = plc.concatenate.concatenate(views)

        result_chunk = TableChunk.from_pylibcudf_table(
            result_table, concat_stream, exclusive_view=True
        )

        await ch_out.send(ctx, Message(0, result_chunk))

    await ch_out.drain(ctx)
