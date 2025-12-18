# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Join and shuffle operators for NDS-H streaming queries."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import pylibcudf as plc
from pylibcudf.contiguous_split import pack

from rapidsmpf.examples.streaming.ndsh.utils import to_device
from rapidsmpf.integrations.cudf.partition import (
    unpack_and_concat as integrations_unpack,
)
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.coll.shuffler import ShufflerAsync
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


class KeepKeys(Enum):
    """Treatment of keys in the result of a join."""

    NO = False
    YES = True


@define_py_node()
async def broadcast(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    op_id: int,
    *,
    ordered: bool = True,
) -> None:
    """
    Broadcast the concatenation of all input messages to all ranks.

    Receives all input chunks, gathers from all ranks, and then provides
    concatenated output.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Input channel of TableChunks.
    ch_out
        Output channel of a single TableChunk.
    op_id
        Disambiguating tag for allgather.
    ordered
        Should the concatenated output be ordered.
    """
    comm = ctx.comm()
    br = ctx.br()

    if comm.nranks == 1:
        # Single rank: just concatenate locally
        chunks: list[TableChunk] = []
        views: list[plc.Table] = []
        gather_stream = ctx.get_stream_from_pool()

        msg: Message[TableChunk] | None
        while (msg := await ch_in.recv(ctx)) is not None:
            chunk = TableChunk.from_message(msg)
            chunk = to_device(ctx, chunk)
            views.append(chunk.table_view())
            chunks.append(chunk)

        if len(chunks) == 1:
            result_chunk = chunks[0]
            await ch_out.send(ctx, Message(0, result_chunk))
        elif len(chunks) == 0:
            empty_table = plc.Table([])
            result_chunk = TableChunk.from_pylibcudf_table(
                empty_table, gather_stream, exclusive_view=True
            )
            await ch_out.send(ctx, Message(0, result_chunk))
        else:
            result_table = plc.concatenate.concatenate(views)
            result_chunk = TableChunk.from_pylibcudf_table(
                result_table, gather_stream, exclusive_view=True
            )
            await ch_out.send(ctx, Message(0, result_chunk))
    else:
        # Multi-rank: use AllGather
        gatherer = AllGather(ctx, op_id)
        stream = ctx.get_stream_from_pool()

        recv_msg: Message[TableChunk] | None
        while (recv_msg := await ch_in.recv(ctx)) is not None:
            chunk = TableChunk.from_message(recv_msg)
            chunk = to_device(ctx, chunk)
            packed = pack(chunk.table_view())
            packed_data = PackedData.from_cudf_packed_columns(packed, stream, br)
            gatherer.insert(recv_msg.sequence_number, packed_data)

        gatherer.insert_finished()
        result_packed = await gatherer.extract_all(ctx, ordered=ordered)

        if len(result_packed) == 1:
            # Single packed result
            unpacked = integrations_unpack([result_packed[0]], stream, br)
            result_chunk = TableChunk.from_pylibcudf_table(
                unpacked, stream, exclusive_view=True
            )
            await ch_out.send(ctx, Message(0, result_chunk))
        else:
            # Multiple packed results - unpack and concatenate
            unpacked = integrations_unpack(result_packed, stream, br)
            result_chunk = TableChunk.from_pylibcudf_table(
                unpacked, stream, exclusive_view=True
            )
            await ch_out.send(ctx, Message(0, result_chunk))

    await ch_out.drain(ctx)


@define_py_node()
async def shuffle(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    keys: list[int],
    num_partitions: int,
    op_id: int,
) -> None:
    """
    Shuffle the input channel by hash-partitioning on given key columns.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Channel of TableChunks to shuffle.
    ch_out
        Channel of shuffled TableChunks.
    keys
        Indices of key columns to shuffle on.
    num_partitions
        Number of output partitions of the shuffle.
    op_id
        Disambiguating tag for the shuffle.
    """
    from rapidsmpf.integrations.cudf.partition import partition_and_pack

    br = ctx.br()
    shuffler = ShufflerAsync(ctx, op_id, num_partitions)

    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        stream = chunk.stream

        # Partition and pack
        packed = partition_and_pack(
            chunk.table_view(),
            keys,
            num_partitions,
            stream,
            br,
        )
        shuffler.insert(packed)

    await shuffler.insert_finished(ctx)

    # Extract local partitions
    comm = ctx.comm()
    for pid in range(num_partitions):
        if pid % comm.nranks == comm.rank:
            result = await shuffler.extract_async(ctx, pid)
            if result is not None:
                stream = ctx.get_stream_from_pool()
                unpacked = integrations_unpack(result, stream, br)
                result_chunk = TableChunk.from_pylibcudf_table(
                    unpacked, stream, exclusive_view=True
                )
                await ch_out.send(ctx, Message(pid, result_chunk))

    await ch_out.drain(ctx)


@define_py_node()
async def left_semi_join_broadcast_left(
    ctx: Context,
    left: Channel[TableChunk],
    right: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    left_on: list[int],
    right_on: list[int],
    op_id: int,
    keep_keys: KeepKeys = KeepKeys.YES,
) -> None:
    """
    Perform a streaming left semi join between two tables.

    This performs a broadcast join, broadcasting the table represented by the
    `left` channel to all ranks, and then streaming through the chunks of the
    `right` channel. The `right` channel is required to provide hash-partitioned
    data in-order.

    Parameters
    ----------
    ctx
        Streaming context.
    left
        Channel of TableChunks used as the broadcasted build side.
    right
        Channel of TableChunks joined in turn against the build side.
    ch_out
        Output channel of TableChunks.
    left_on
        Column indices of the keys in the left table.
    right_on
        Column indices of the keys in the right table.
    op_id
        Disambiguating tag for the broadcast of the left table.
    keep_keys
        Does the result contain the key columns, or only "carrier" value columns.
    """
    # First, broadcast and gather the left table
    left_chunks: list[TableChunk] = []
    left_views: list[plc.Table] = []
    gather_stream = ctx.get_stream_from_pool()

    # Collect all left chunks
    comm = ctx.comm()
    br = ctx.br()

    if comm.nranks == 1:
        left_msg: Message[TableChunk] | None
        while (left_msg := await left.recv(ctx)) is not None:
            chunk = TableChunk.from_message(left_msg)
            chunk = to_device(ctx, chunk)
            left_views.append(chunk.table_view())
            left_chunks.append(chunk)

        if len(left_chunks) == 0:
            left_table_view = plc.Table([])
        elif len(left_chunks) == 1:
            left_table_view = left_views[0]
        else:
            left_table_view = plc.concatenate.concatenate(left_views)
    else:
        # Multi-rank: use AllGather
        gatherer = AllGather(ctx, op_id)

        left_recv_msg: Message[TableChunk] | None
        while (left_recv_msg := await left.recv(ctx)) is not None:
            chunk = TableChunk.from_message(left_recv_msg)
            chunk = to_device(ctx, chunk)
            packed = pack(chunk.table_view())
            packed_data = PackedData.from_cudf_packed_columns(packed, gather_stream, br)
            gatherer.insert(left_recv_msg.sequence_number, packed_data)

        gatherer.insert_finished()
        result_packed = await gatherer.extract_all(ctx, ordered=True)

        if len(result_packed) == 0:
            left_table_view = plc.Table([])
        else:
            left_table_view = integrations_unpack(result_packed, gather_stream, br)

    # Now stream through the right channel and perform semi-join
    sequence = 0

    right_msg: Message[TableChunk] | None
    while (right_msg := await right.recv(ctx)) is not None:
        right_chunk = TableChunk.from_message(right_msg)
        right_chunk = to_device(ctx, right_chunk)
        right_stream = right_chunk.stream
        right_table = right_chunk.table_view()

        # Build probe/build tables for semi join
        left_key_columns = [left_table_view.columns()[i] for i in left_on]
        left_keys = plc.Table(left_key_columns)

        right_key_columns = [right_table.columns()[i] for i in right_on]
        right_keys = plc.Table(right_key_columns)

        # Perform left semi join
        # In a semi-join, we return rows from left that have matches in right
        # TODO: streams?
        left_indices = plc.join.left_semi_join(
            left_keys, right_keys, plc.types.NullEquality.UNEQUAL
        )

        # Gather the matching rows from left table
        result_columns: list[plc.Column] = []
        for i in range(left_table_view.num_columns()):
            col = left_table_view.columns()[i]
            gathered = plc.copying.gather(
                plc.Table([col]),
                left_indices,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            result_columns.append(gathered.columns()[0])

        result_table = plc.Table(result_columns)

        result_chunk = TableChunk.from_pylibcudf_table(
            result_table, right_stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(sequence, result_chunk))
        sequence += 1

    await ch_out.drain(ctx)


@define_py_node()
async def left_semi_join_shuffle(
    ctx: Context,
    left: Channel[TableChunk],
    right: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    left_on: list[int],
    right_on: list[int],
) -> None:
    """
    Perform a streaming left semi join between two shuffled tables.

    This performs a shuffle join. The left and right channels are required to
    provide hash-partitioned data in matching order.

    Parameters
    ----------
    ctx
        Streaming context.
    left
        Channel of TableChunks in hash-partitioned order.
    right
        Channel of TableChunks in matching hash-partitioned order.
    ch_out
        Output channel of TableChunks.
    left_on
        Column indices of the keys in the left table.
    right_on
        Column indices of the keys in the right table.
    """
    while True:
        # Requirement: two shuffles kick out partitions in the same order
        left_msg = await left.recv(ctx)
        right_msg = await right.recv(ctx)

        if left_msg is None:
            if right_msg is not None:
                raise RuntimeError(
                    "Left does not have same number of partitions as right"
                )
            break

        if right_msg is None:
            raise RuntimeError("Left does not have same number of partitions as right")

        if left_msg.sequence_number != right_msg.sequence_number:
            raise RuntimeError("Mismatching sequence numbers")

        left_chunk = TableChunk.from_message(left_msg)
        left_chunk = to_device(ctx, left_chunk)
        left_table = left_chunk.table_view()

        right_chunk = TableChunk.from_message(right_msg)
        right_chunk = to_device(ctx, right_chunk)
        right_stream = right_chunk.stream
        right_table = right_chunk.table_view()

        # Build key tables
        left_key_columns = [left_table.columns()[i] for i in left_on]
        left_keys = plc.Table(left_key_columns)

        right_key_columns = [right_table.columns()[i] for i in right_on]
        right_keys = plc.Table(right_key_columns)

        # Perform left semi join
        left_indices = plc.join.left_semi_join(
            left_keys, right_keys, plc.types.NullEquality.UNEQUAL
        )

        # Gather the matching rows from left table
        result_columns: list[plc.Column] = []
        for i in range(left_table.num_columns()):
            col = left_table.columns()[i]
            gathered = plc.copying.gather(
                plc.Table([col]),
                left_indices,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            )
            result_columns.append(gathered.columns()[0])

        result_table = plc.Table(result_columns)

        result_chunk = TableChunk.from_pylibcudf_table(
            result_table, right_stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(left_msg.sequence_number, result_chunk))

    await ch_out.drain(ctx)
