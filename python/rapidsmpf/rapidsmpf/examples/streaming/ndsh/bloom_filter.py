# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Bloom filter operations for NDS-H streaming queries.

This module provides a simple bloom filter implementation for approximate
set membership queries, used to pre-filter tables before expensive joins.

Note: This implementation uses pylibcudf operations to ensure proper
stream and memory resource handling.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pylibcudf as plc
from pylibcudf.contiguous_split import pack

from rapidsmpf.examples.streaming.ndsh.utils import to_device
from rapidsmpf.memory.packed_data import PackedData
from rapidsmpf.streaming.coll.allgather import AllGather
from rapidsmpf.streaming.core.message import Message
from rapidsmpf.streaming.core.node import define_py_node
from rapidsmpf.streaming.cudf.table_chunk import TableChunk

if TYPE_CHECKING:
    from rapidsmpf.streaming.core.channel import Channel
    from rapidsmpf.streaming.core.context import Context


class BloomFilter:
    """
    A bloom filter for approximate set membership queries.

    This implementation uses pylibcudf operations for GPU computation,
    ensuring proper stream and memory resource handling.

    Parameters
    ----------
    num_blocks
        Number of blocks in the filter.
    seed
        Seed used for hashing each value.
    """

    # Bytes per block (512 bits = 64 bytes)
    BYTES_PER_BLOCK = 64

    def __init__(self, num_blocks: int, seed: int = 0):
        self.num_blocks = num_blocks
        self.seed = seed
        # Initialize storage as pylibcudf Column of zeros (uint8)
        size = num_blocks * self.BYTES_PER_BLOCK
        zero_scalar = plc.Scalar.from_py(0, plc.DataType(plc.TypeId.UINT8))
        self._storage: plc.Column = plc.Column.from_scalar(zero_scalar, size)

    @staticmethod
    def fitting_num_blocks(l2_cache_size: int) -> int:
        """
        Return number of blocks to use if the filter should fit in a given L2 cache size.

        Parameters
        ----------
        l2_cache_size
            Size of the L2 cache in bytes.

        Returns
        -------
        int
            Number of blocks that fit in the cache.
        """
        return max(1, l2_cache_size // BloomFilter.BYTES_PER_BLOCK)

    def add(self, table: plc.Table) -> None:
        """
        Add values to the filter.

        Parameters
        ----------
        table
            Table of values to hash and add to the filter.
        """
        if table.num_rows() == 0:
            return

        # Hash the table using xxhash64 to get int64 hashes
        hashes = plc.hashing.xxhash_64(table, self.seed)

        filter_bits = self.num_blocks * self.BYTES_PER_BLOCK * 8

        # Compute bit_positions = hashes % filter_bits
        filter_bits_scalar = plc.Scalar.from_py(
            filter_bits, plc.DataType(plc.TypeId.INT64)
        )
        bit_positions = plc.binaryop.binary_operation(
            hashes,
            filter_bits_scalar,
            plc.binaryop.BinaryOperator.MOD,
            plc.DataType(plc.TypeId.INT64),
        )

        # Compute byte_positions = bit_positions // 8
        eight_scalar = plc.Scalar.from_py(8, plc.DataType(plc.TypeId.INT64))
        byte_positions = plc.binaryop.binary_operation(
            bit_positions,
            eight_scalar,
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            plc.DataType(plc.TypeId.INT64),
        )

        # Compute bit_offsets = bit_positions % 8
        bit_offsets = plc.binaryop.binary_operation(
            bit_positions,
            eight_scalar,
            plc.binaryop.BinaryOperator.MOD,
            plc.DataType(plc.TypeId.INT64),
        )

        # Remove duplicate (byte_pos, bit_offset) pairs so sum works as OR
        pairs = plc.Table([byte_positions, bit_offsets])
        distinct_pairs = plc.stream_compaction.distinct(
            pairs,
            [0, 1],  # keys: both columns
            plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
        )

        distinct_byte_positions = distinct_pairs.columns()[0]
        distinct_bit_offsets = distinct_pairs.columns()[1]

        # Cast bit_offsets to uint8 for shift operation
        distinct_bit_offsets_u8 = plc.unary.cast(
            distinct_bit_offsets, plc.DataType(plc.TypeId.UINT8)
        )

        # Compute masks = 1 << bit_offsets
        one_scalar = plc.Scalar.from_py(1, plc.DataType(plc.TypeId.UINT8))
        one_column = plc.Column.from_scalar(one_scalar, distinct_pairs.num_rows())
        masks = plc.binaryop.binary_operation(
            one_column,
            distinct_bit_offsets_u8,
            plc.binaryop.BinaryOperator.SHIFT_LEFT,
            plc.DataType(plc.TypeId.UINT8),
        )

        # Groupby byte_positions, sum masks
        # For distinct single-bit masks, sum equals bitwise OR
        grouper = plc.groupby.GroupBy(
            plc.Table([distinct_byte_positions]),
            plc.types.NullPolicy.INCLUDE,
        )
        agg_request = plc.groupby.GroupByRequest(masks, [plc.aggregation.sum()])
        grouped_keys, grouped_results = grouper.aggregate([agg_request])

        # Get unique byte positions and their aggregated masks
        unique_positions = grouped_keys.columns()[0]
        aggregated_masks = grouped_results[0].columns()[0]

        # Cast aggregated masks back to uint8 (sum may produce larger type)
        aggregated_masks_u8 = plc.unary.cast(
            aggregated_masks, plc.DataType(plc.TypeId.UINT8)
        )

        # Cast positions to int32 for gather/scatter operations
        indices = plc.unary.cast(unique_positions, plc.DataType(plc.TypeId.INT32))

        # Gather current values from storage at these positions
        current_values = plc.copying.gather(
            plc.Table([self._storage]),
            indices,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # OR current values with new masks
        new_values = plc.binaryop.binary_operation(
            current_values,
            aggregated_masks_u8,
            plc.binaryop.BinaryOperator.BITWISE_OR,
            plc.DataType(plc.TypeId.UINT8),
        )

        # Scatter back to storage
        self._storage = plc.copying.scatter(
            plc.Table([new_values]),
            indices,
            plc.Table([self._storage]),
        ).columns()[0]

    def merge(self, other: BloomFilter) -> None:
        """
        Merge another filter into this one, computing their union.

        Parameters
        ----------
        other
            Other filter to merge into this one.

        Raises
        ------
        ValueError
            If `other` is not compatible with this filter.
        """
        if self.num_blocks != other.num_blocks:
            raise ValueError("Cannot merge filters with different number of blocks")
        if self.seed != other.seed:
            raise ValueError("Cannot merge filters with different seeds")

        # Bitwise OR the storage columns
        self._storage = plc.binaryop.binary_operation(
            self._storage,
            other._storage,
            plc.binaryop.BinaryOperator.BITWISE_OR,
            plc.DataType(plc.TypeId.UINT8),
        )

    def contains(self, table: plc.Table) -> plc.Column:
        """
        Return a mask of which rows might be contained in the filter.

        Parameters
        ----------
        table
            Values to check for set membership.

        Returns
        -------
        plc.Column
            Boolean column where True indicates the value might be in the set.
        """
        if table.num_rows() == 0:
            # Return empty boolean column
            return plc.Column.from_scalar(
                plc.Scalar.from_py(py_val=True, dtype=plc.DataType(plc.TypeId.BOOL8)), 0
            )

        # Hash the table
        hashes = plc.hashing.xxhash_64(table, self.seed)

        filter_bits = self.num_blocks * self.BYTES_PER_BLOCK * 8

        # Compute bit_positions = hashes % filter_bits
        filter_bits_scalar = plc.Scalar.from_py(
            filter_bits, plc.DataType(plc.TypeId.INT64)
        )
        bit_positions = plc.binaryop.binary_operation(
            hashes,
            filter_bits_scalar,
            plc.binaryop.BinaryOperator.MOD,
            plc.DataType(plc.TypeId.INT64),
        )

        # Compute byte_positions = bit_positions // 8
        eight_scalar = plc.Scalar.from_py(8, plc.DataType(plc.TypeId.INT64))
        byte_positions = plc.binaryop.binary_operation(
            bit_positions,
            eight_scalar,
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            plc.DataType(plc.TypeId.INT64),
        )

        # Compute bit_offsets = bit_positions % 8
        bit_offsets = plc.binaryop.binary_operation(
            bit_positions,
            eight_scalar,
            plc.binaryop.BinaryOperator.MOD,
            plc.DataType(plc.TypeId.INT64),
        )

        # Cast positions to int32 for gather
        indices = plc.unary.cast(byte_positions, plc.DataType(plc.TypeId.INT32))

        # Gather byte values from storage
        byte_values = plc.copying.gather(
            plc.Table([self._storage]),
            indices,
            plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        ).columns()[0]

        # Cast bit_offsets to uint8 for shift
        bit_offsets_u8 = plc.unary.cast(bit_offsets, plc.DataType(plc.TypeId.UINT8))

        # Compute masks = 1 << bit_offsets
        one_scalar = plc.Scalar.from_py(1, plc.DataType(plc.TypeId.UINT8))
        one_column = plc.Column.from_scalar(one_scalar, table.num_rows())
        masks = plc.binaryop.binary_operation(
            one_column,
            bit_offsets_u8,
            plc.binaryop.BinaryOperator.SHIFT_LEFT,
            plc.DataType(plc.TypeId.UINT8),
        )

        # Check if bits are set: (byte_values & masks) != 0
        and_result = plc.binaryop.binary_operation(
            byte_values,
            masks,
            plc.binaryop.BinaryOperator.BITWISE_AND,
            plc.DataType(plc.TypeId.UINT8),
        )

        zero_scalar = plc.Scalar.from_py(0, plc.DataType(plc.TypeId.UINT8))
        return plc.binaryop.binary_operation(
            and_result,
            zero_scalar,
            plc.binaryop.BinaryOperator.NOT_EQUAL,
            plc.DataType(plc.TypeId.BOOL8),
        )

    @property
    def data(self) -> plc.Column:
        """Return the underlying storage column."""
        return self._storage

    @property
    def size(self) -> int:
        """Return size in bytes of the underlying storage."""
        return self._storage.size()  # type: ignore[no-any-return]


class BloomFilterChunk:
    """
    A wrapper to hold a BloomFilter as a channel payload.

    Parameters
    ----------
    bloom_filter
        The bloom filter to wrap.
    """

    def __init__(self, bloom_filter: BloomFilter):
        self.bloom_filter = bloom_filter


@define_py_node()
async def build_bloom_filter(
    ctx: Context,
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    op_id: int,
    seed: int = 0,
    num_filter_blocks: int = 1024,
) -> None:
    """
    Build a bloom filter of the input channel.

    Parameters
    ----------
    ctx
        Streaming context.
    ch_in
        Input channel of TableChunks to build bloom filter for.
    ch_out
        Output channel receiving a single message containing the bloom filter.
        The bloom filter is wrapped in a TableChunk containing a single uint8
        column representing the filter data.
    op_id
        Disambiguating tag to combine filters across ranks.
    seed
        Hash seed for hashing the keys.
    num_filter_blocks
        Number of blocks in the filter.
    """
    comm = ctx.comm()
    br = ctx.br()
    stream = ctx.get_stream_from_pool()

    # Build local bloom filter
    local_filter = BloomFilter(num_filter_blocks, seed)

    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        table = chunk.table_view()
        local_filter.add(table)

    if comm.nranks == 1:
        # Single rank - just output the filter as a table
        filter_table = plc.Table([local_filter.data])
        result_chunk = TableChunk.from_pylibcudf_table(
            filter_table, stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(0, result_chunk))
    else:
        # Multi-rank: gather and merge filters
        filter_table = plc.Table([local_filter.data])

        packed = pack(filter_table)
        packed_data = PackedData.from_cudf_packed_columns(packed, stream, br)

        gatherer = AllGather(ctx, op_id)
        gatherer.insert(0, packed_data)
        gatherer.insert_finished()

        result_packed = await gatherer.extract_all(ctx, ordered=False)

        # Merge all filters
        merged_filter = BloomFilter(num_filter_blocks, seed)
        from rapidsmpf.integrations.cudf.partition import (
            unpack_and_concat as integrations_unpack,
        )

        for pd in result_packed:
            unpacked = integrations_unpack([pd], stream, br)
            other_filter = BloomFilter(num_filter_blocks, seed)
            other_filter._storage = unpacked.columns()[0]
            merged_filter.merge(other_filter)

        # Output the merged filter
        filter_table = plc.Table([merged_filter.data])
        result_chunk = TableChunk.from_pylibcudf_table(
            filter_table, stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(0, result_chunk))

    await ch_out.drain(ctx)


@define_py_node()
async def apply_bloom_filter(
    ctx: Context,
    bloom_filter_ch: Channel[TableChunk],
    ch_in: Channel[TableChunk],
    ch_out: Channel[TableChunk],
    keys: list[int],
    seed: int = 0,
) -> None:
    """
    Apply a bloom filter to an input channel.

    Parameters
    ----------
    ctx
        Streaming context.
    bloom_filter_ch
        Channel containing the bloom filter (a single message).
        The filter is expected as a TableChunk with a single uint8 column.
    ch_in
        Input channel of TableChunks to apply bloom filter to.
    ch_out
        Output channel receiving filtered TableChunks.
    keys
        Indices selecting the key columns for the hash fingerprint.
    seed
        Hash seed used when building the filter.
    """
    # First, receive the bloom filter
    filter_msg = await bloom_filter_ch.recv(ctx)
    if filter_msg is None:
        raise RuntimeError("Expected bloom filter message")

    filter_chunk = TableChunk.from_message(filter_msg)
    filter_chunk = to_device(ctx, filter_chunk)
    filter_table = filter_chunk.table_view()

    # Reconstruct the bloom filter from the table
    filter_data = filter_table.columns()[0]
    num_blocks = filter_data.size() // BloomFilter.BYTES_PER_BLOCK
    bloom_filter = BloomFilter(num_blocks, seed)
    bloom_filter._storage = filter_data

    # Now filter incoming chunks
    msg: Message[TableChunk] | None
    while (msg := await ch_in.recv(ctx)) is not None:
        chunk = TableChunk.from_message(msg)
        chunk = to_device(ctx, chunk)
        stream = chunk.stream
        table = chunk.table_view()

        # Select key columns for probing
        key_columns = [table.columns()[k] for k in keys]
        key_table = plc.Table(key_columns)

        # Check which rows might be in the filter
        mask = bloom_filter.contains(key_table)

        # Apply boolean mask to filter rows
        filtered_table = plc.stream_compaction.apply_boolean_mask(table, mask)

        result_chunk = TableChunk.from_pylibcudf_table(
            filtered_table, stream, exclusive_view=True
        )
        await ch_out.send(ctx, Message(msg.sequence_number, result_chunk))

    await ch_out.drain(ctx)
