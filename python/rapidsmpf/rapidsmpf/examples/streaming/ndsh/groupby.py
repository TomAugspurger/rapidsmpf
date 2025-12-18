# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Groupby operators for NDS-H streaming queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import pylibcudf as plc

from rapidsmpf.examples.streaming.ndsh.utils import to_device
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


class GroupbyRequest(NamedTuple):
    """Description of aggregation requests on a given column."""

    column_idx: int
    """Index of column in input table to aggregate."""

    aggregations: list[Callable[[], plc.aggregation.Aggregation]]
    """Functions to generate aggregations to perform on the column."""


@define_py_node()
async def chunkwise_group_by(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    keys: list[int],
    requests: list[GroupbyRequest],
    *,
    include_nulls: bool = True,
) -> None:
    """
    Perform a chunkwise grouped aggregation.

    Grouped chunks are not further grouped together.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        TableChunks to aggregate.
    ch_out
        Output channel of grouped TableChunks.
    keys
        Column indices of the key columns in the input channel.
    requests
        List of aggregation requests referencing columns in the input channel.
    include_nulls
        How nulls in the key columns are treated. If True, include nulls.
    """
    null_policy = (
        plc.types.NullPolicy.INCLUDE if include_nulls else plc.types.NullPolicy.EXCLUDE
    )

    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk, allow_overbooking=True)
        stream = chunk.stream
        table = chunk.table_view()

        # Build aggregation requests for pylibcudf
        agg_requests: list[plc.groupby.GroupByRequest] = []
        for req in requests:
            column = table.columns()[req.column_idx]
            aggs = [agg_fn() for agg_fn in req.aggregations]
            agg_requests.append(plc.groupby.GroupByRequest(column, aggs))

        # Select key columns
        key_columns = [table.columns()[k] for k in keys]
        key_table = plc.Table(key_columns)

        # Perform groupby
        grouper = plc.groupby.GroupBy(key_table, null_policy)
        grouped_keys, grouped_results = grouper.aggregate(agg_requests)

        # Build result table: keys + aggregated values
        result_columns: list[plc.Column] = list(grouped_keys.columns())
        for result in grouped_results:
            result_columns.extend(result.columns())

        result_table = plc.Table(result_columns)

        result_chunk = TableChunk.from_pylibcudf_table(
            result_table, stream, exclusive_view=True
        )

        await ch_out.send(ctx, Message(msg.sequence_number, result_chunk))

    await ch_out.drain(ctx)
