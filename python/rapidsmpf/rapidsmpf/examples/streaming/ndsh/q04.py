# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
TPC-H Query 4 implementation using rapidsmpf streaming API.

The SQL form of the query is:

.. code-block:: sql

    SELECT
        o_orderpriority,
        count(*) as order_count
    FROM
        orders
    WHERE
        o_orderdate >= TIMESTAMP '1993-07-01'
        AND o_orderdate < TIMESTAMP '1993-07-01' + INTERVAL '3' MONTH
        AND EXISTS (
            SELECT
                *
            FROM
                lineitem
            WHERE
                l_orderkey = o_orderkey
                AND l_commitdate < l_receiptdate
        )
    GROUP BY
        o_orderpriority
    ORDER BY
        o_orderpriority

The "exists" clause is translated into a left-semi join in libcudf.
"""

from __future__ import annotations

import datetime
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pylibcudf as plc
import rmm.mr

from rapidsmpf.examples.streaming.ndsh.bloom_filter import (
    BloomFilter,
    apply_bloom_filter,
    build_bloom_filter,
)
from rapidsmpf.examples.streaming.ndsh.concatenate import ConcatOrder, concatenate
from rapidsmpf.examples.streaming.ndsh.groupby import GroupbyRequest, chunkwise_group_by
from rapidsmpf.examples.streaming.ndsh.join import (
    KeepKeys,
    broadcast,
    left_semi_join_broadcast_left,
    left_semi_join_shuffle,
    shuffle,
)
from rapidsmpf.examples.streaming.ndsh.parquet_writer import write_parquet
from rapidsmpf.examples.streaming.ndsh.utils import (
    create_context,
    get_table_path,
    list_parquet_files,
    parse_arguments,
    sink_channel,
    to_device,
)
from rapidsmpf.rmm_resource_adaptor import RmmResourceAdaptor
from rapidsmpf.streaming.core.fanout import FanoutPolicy, fanout
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import (
    define_py_node,
    run_streaming_pipeline,
)
from rapidsmpf.streaming.cudf.parquet import Filter, read_parquet
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from rapidsmpf.examples.streaming.ndsh.utlis import ProgramOptions
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context
    from rapidsmpf.streaming.core.node import CppNode, PyNode


def chunkwise_groupby_requests() -> list[GroupbyRequest]:
    """Return groupby aggregation requests for chunkwise groupby."""
    return [
        GroupbyRequest(
            column_idx=0,  # Count on first column (o_orderpriority after projection)
            aggregations=[lambda: plc.aggregation.count(plc.types.NullPolicy.INCLUDE)],
        )
    ]


@define_py_node()
async def final_groupby_agg(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Perform final groupby aggregation.

    Since the cardinality of o_orderpriority is very low, the chunkwise groupby
    produces a set of small tables that fits comfortably in memory. We can perform
    regular cudf groupby operations instead of streaming versions.

    Input table:
        - o_orderpriority
        - order_count (partial counts from chunkwise groupby)

    Output table:
        - o_orderpriority
        - order_count (sum of partial counts)
    """
    msg = await ch_in.recv(ctx)
    if msg is None:
        raise RuntimeError("Expected input for final groupby")

    # Verify no more input
    next_msg = await ch_in.recv(ctx)
    if next_msg is not None:
        raise RuntimeError("Expecting concatenated input at this point")

    chunk = TableChunk.from_message(msg)
    chunk = to_device(ctx, chunk)
    stream = chunk.stream
    table = chunk.table_view()

    # Perform groupby sum aggregation
    key_table = plc.Table([table.columns()[0]])
    grouper = plc.groupby.GroupBy(key_table, plc.types.NullPolicy.INCLUDE)

    # Sum aggregation on the count column
    sum_agg = plc.aggregation.sum()
    request = plc.groupby.GroupByRequest(table.columns()[1], [sum_agg])

    grouped_keys, grouped_results = grouper.aggregate([request])

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


@define_py_node()
async def sort_by(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Sort the grouped orders table by o_orderpriority.

    Input table:
        - o_orderpriority
        - order_count

    Output table:
        - o_orderpriority (sorted ascending)
        - order_count
    """
    msg = await ch_in.recv(ctx)
    if msg is None:
        return

    chunk = TableChunk.from_message(msg)
    chunk = to_device(ctx, chunk)
    stream = chunk.stream
    table = chunk.table_view()

    # Sort by first column (o_orderpriority) ascending
    key_table = plc.Table([table.columns()[0]])
    sorted_table = plc.sorting.sort_by_key(
        table,
        key_table,
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.BEFORE],
    )

    result_chunk = TableChunk.from_pylibcudf_table(
        sorted_table, stream, exclusive_view=True
    )
    await ch_out.send(ctx, Message(msg.sequence_number, result_chunk))
    await ch_out.drain(ctx)


@define_py_node()
async def select_columns(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    indices: list[int],
) -> None:
    """
    Select specific columns from the table.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Input channel.
    ch_out
        Output channel.
    indices
        Column indices to select.
    """
    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        stream = chunk.stream
        table = chunk.table_view()

        # Select specified columns
        cols = table.columns()
        selected_columns = [cols[i] for i in indices]
        result_table = plc.Table(selected_columns)

        result_chunk = TableChunk.from_pylibcudf_table(
            result_table, stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(msg.sequence_number, result_chunk))

    await ch_out.drain(ctx)


@define_py_node()
async def filter_lineitem(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
) -> None:
    """
    Filter lineitem on l_commitdate < l_receiptdate.

    Input table:
        - l_commitdate (column 0)
        - l_receiptdate (column 1)
        - l_orderkey (column 2)

    Output table:
        - l_orderkey
    """
    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        stream = chunk.stream
        table = chunk.table_view()

        l_commitdate = table.columns()[0]
        l_receiptdate = table.columns()[1]

        # l_commitdate < l_receiptdate
        mask = plc.binaryop.binary_operation(
            l_commitdate,
            l_receiptdate,
            plc.binaryop.BinaryOperator.LESS,
            plc.types.DataType(plc.types.TypeId.BOOL8),
        )

        # Select only l_orderkey column and apply mask
        orderkey_table = plc.Table([table.columns()[2]])
        filtered_table = plc.stream_compaction.apply_boolean_mask(orderkey_table, mask)

        result_chunk = TableChunk.from_pylibcudf_table(
            filtered_table, stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(msg.sequence_number, result_chunk))

    await ch_out.drain(ctx)


def make_read_lineitem_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    num_producers: int,
    num_rows_per_chunk: int,
    input_directory: str,
) -> CppNode:
    """
    Create a node to read the lineitem table.

    Returns a CppNode that reads columns:
        - l_commitdate (used in filter)
        - l_receiptdate (used in filter)
        - l_orderkey (used in join)
    """
    files = list_parquet_files(get_table_path(input_directory, "lineitem"))

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(files)
    ).build()
    options.set_columns(
        [
            "l_commitdate",
            "l_receiptdate",
            "l_orderkey",
        ]
    )

    return read_parquet(
        ctx,
        ch_out,
        num_producers,
        options,
        num_rows_per_chunk,
    )


def make_read_orders_node(
    ctx: Context,
    ch_out: Channel[TableChunk],
    num_producers: int,
    num_rows_per_chunk: int,
    input_directory: str,
) -> CppNode:
    """
    Create a node to read the orders table with date filter.

    Returns a CppNode that reads columns:
        - o_orderkey (used in join)
        - o_orderpriority (used in group by)

    With filter: 1993-07-01 <= o_orderdate < 1993-10-01
    """
    files = list_parquet_files(get_table_path(input_directory, "orders"))

    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.SourceInfo(files)
    ).build()
    options.set_columns(
        [
            "o_orderkey",
            "o_orderpriority",
        ]
    )

    # Build the filter expression: 1993-07-01 <= o_orderdate < 1993-10-01
    stream = ctx.get_stream_from_pool()

    # Create timestamp literals
    ts1 = datetime.datetime(1993, 7, 1)
    ts2 = datetime.datetime(1993, 10, 1)

    # Build filter expression using pylibcudf expressions
    col_ref = plc.expressions.ColumnNameReference("o_orderdate")
    # (datetime.datetime(2020, 1, 1), TypeId.TIMESTAMP_MILLISECONDS),

    scalar1 = plc.Scalar.from_py(
        ts1, plc.types.DataType(plc.types.TypeId.TIMESTAMP_MILLISECONDS)
    )
    scalar2 = plc.Scalar.from_py(
        ts2, plc.types.DataType(plc.types.TypeId.TIMESTAMP_MILLISECONDS)
    )

    literal1 = plc.expressions.Literal(scalar1)
    literal2 = plc.expressions.Literal(scalar2)

    # o_orderdate >= ts1
    ge_expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.GREATER_EQUAL,
        col_ref,
        literal1,
    )

    # o_orderdate < ts2
    lt_expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.LESS,
        col_ref,
        literal2,
    )

    # AND them together
    and_expr = plc.expressions.Operation(
        plc.expressions.ASTOperator.LOGICAL_AND,
        ge_expr,
        lt_expr,
    )

    filter_obj = Filter(stream, and_expr)

    return read_parquet(
        ctx,
        ch_out,
        num_producers,
        options,
        num_rows_per_chunk,
        filter_obj,
    )


def run_q04(options: ProgramOptions) -> None:
    """Run TPC-H Query 4."""
    # Initialize RMM
    mr = RmmResourceAdaptor(rmm.mr.get_current_device_resource())
    ctx = create_context(options, mr)

    # Executor for Python nodes
    py_executor = ThreadPoolExecutor(max_workers=options.num_streaming_threads)

    # Get L2 cache size for bloom filter sizing
    # Default to 40MB if not available
    l2_cache_size = 40 * 1024 * 1024
    num_filter_blocks = BloomFilter.fitting_num_blocks(l2_cache_size)

    timings: list[float] = []

    for iteration in range(options.num_iterations):
        op_id = 0
        nodes: list[CppNode | PyNode] = []

        start = time.perf_counter()

        # === Pipeline Construction ===

        # Lineitem Table: [l_commitdate, l_receiptdate, l_orderkey]
        lineitem: Channel[TableChunk] = ctx.create_channel()
        # Filtered lineitem: [l_orderkey]
        filtered_lineitem: Channel[TableChunk] = ctx.create_channel()
        # Shuffled filtered lineitem: [l_orderkey]
        filtered_lineitem_shuffled: Channel[TableChunk] = ctx.create_channel()

        # Orders Table: [o_orderkey, o_orderpriority]
        order: Channel[TableChunk] = ctx.create_channel()

        # Joined orders x lineitem: [o_orderkey, o_orderpriority]
        orders_x_lineitem: Channel[TableChunk] = ctx.create_channel()

        # Projected columns: [o_orderpriority]
        projected_columns: Channel[TableChunk] = ctx.create_channel()
        # Grouped chunkwise: [o_orderpriority, order_count]
        grouped_chunkwise: Channel[TableChunk] = ctx.create_channel()

        # Read lineitem
        nodes.append(
            make_read_lineitem_node(
                ctx, lineitem, 4, options.num_rows_per_chunk, options.input_directory
            )
        )

        # Filter lineitem
        nodes.append(filter_lineitem(ctx, lineitem, filtered_lineitem))

        # Read orders
        nodes.append(
            make_read_orders_node(
                ctx, order, 4, options.num_rows_per_chunk, options.input_directory
            )
        )

        # Fanout filtered orders: one for bloom filter, one for join
        bloom_filter_input: Channel[TableChunk] = ctx.create_channel()
        orders_for_join: Channel[TableChunk] = ctx.create_channel()

        # Use fanout to split orders to bloom filter and join paths
        nodes.append(
            fanout(
                ctx, order, [bloom_filter_input, orders_for_join], FanoutPolicy.BOUNDED
            )
        )

        # Build bloom filter from filtered orders' o_orderkey
        bloom_filter_output: Channel[TableChunk] = ctx.create_channel()
        nodes.append(
            build_bloom_filter(
                ctx,
                bloom_filter_input,
                bloom_filter_output,
                op_id=10 * iteration + op_id,
                seed=0,
                num_filter_blocks=num_filter_blocks,
            )
        )
        op_id += 1

        # Apply bloom filter to filtered lineitem before shuffling
        bloom_filtered_lineitem: Channel[TableChunk] = ctx.create_channel()
        nodes.append(
            apply_bloom_filter(
                ctx,
                bloom_filter_output,
                filtered_lineitem,
                bloom_filtered_lineitem,
                keys=[0],
                seed=0,
            )
        )

        # Shuffle the filtered lineitem table
        num_partitions = 16
        nodes.append(
            shuffle(
                ctx,
                bloom_filtered_lineitem,
                filtered_lineitem_shuffled,
                keys=[0],
                num_partitions=num_partitions,
                op_id=10 * iteration + op_id,
            )
        )
        op_id += 1

        if options.use_shuffle_join:
            # Shuffle join path
            filtered_order_shuffled: Channel[TableChunk] = ctx.create_channel()
            nodes.append(
                shuffle(
                    ctx,
                    orders_for_join,
                    filtered_order_shuffled,
                    keys=[0],
                    num_partitions=num_partitions,
                    op_id=10 * iteration + op_id,
                )
            )
            op_id += 1

            nodes.append(
                left_semi_join_shuffle(
                    ctx,
                    filtered_order_shuffled,
                    filtered_lineitem_shuffled,
                    orders_x_lineitem,
                    left_on=[0],
                    right_on=[0],
                )
            )
        else:
            # Broadcast join path
            nodes.append(
                left_semi_join_broadcast_left(
                    ctx,
                    orders_for_join,
                    filtered_lineitem_shuffled,
                    orders_x_lineitem,
                    left_on=[0],
                    right_on=[0],
                    op_id=10 * iteration + op_id,
                    keep_keys=KeepKeys.YES,
                )
            )
            op_id += 1

        # Select columns: keep only o_orderpriority (index 1)
        nodes.append(
            select_columns(ctx, orders_x_lineitem, projected_columns, indices=[1])
        )

        # Chunkwise groupby
        nodes.append(
            chunkwise_group_by(
                ctx,
                projected_columns,
                grouped_chunkwise,
                keys=[0],
                requests=chunkwise_groupby_requests(),
                include_nulls=True,
            )
        )

        # Final aggregation
        final_groupby_input: Channel[TableChunk] = ctx.create_channel()
        if ctx.comm().nranks > 1:
            nodes.append(
                broadcast(
                    ctx,
                    grouped_chunkwise,
                    final_groupby_input,
                    op_id=10 * iteration + op_id,
                    ordered=False,
                )
            )
            op_id += 1
        else:
            nodes.append(
                concatenate(
                    ctx, grouped_chunkwise, final_groupby_input, ConcatOrder.DONT_CARE
                )
            )

        if ctx.comm().rank == 0:
            final_groupby_output: Channel[TableChunk] = ctx.create_channel()
            nodes.append(
                final_groupby_agg(ctx, final_groupby_input, final_groupby_output)
            )

            sorted_output: Channel[TableChunk] = ctx.create_channel()
            nodes.append(sort_by(ctx, final_groupby_output, sorted_output))

            nodes.append(
                write_parquet(
                    ctx,
                    sorted_output,
                    options.output_file,
                    column_names=["o_orderpriority", "order_count"],
                )
            )
        else:
            nodes.append(sink_channel(ctx, final_groupby_input))

        pipeline_time = time.perf_counter() - start

        # === Execute Pipeline ===
        start = time.perf_counter()
        run_streaming_pipeline(nodes=nodes, py_executor=py_executor)
        compute_time = time.perf_counter() - start

        timings.extend([pipeline_time, compute_time])

        # Print statistics
        ctx.comm().logger.print(ctx.statistics().report())
        ctx.statistics().clear()

    # Print timing summary
    if ctx.comm().rank == 0:
        for i in range(options.num_iterations):
            ctx.comm().logger.print(
                f"Iteration {i} pipeline construction time [s]: {timings[2 * i]:.4f}"
            )
            ctx.comm().logger.print(
                f"Iteration {i} compute time [s]: {timings[2 * i + 1]:.4f}"
            )


def main(args: list[str] | None = None) -> None:
    """Entry point for TPC-H Query 4."""
    options = parse_arguments(args)
    run_q04(options)


if __name__ == "__main__":
    main()
