"""Geo things."""
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import concurrent.futures
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr

if TYPE_CHECKING:
    from collections.abc import Callable


def parse_shape(value: str) -> tuple[int, int, int]:
    """
    Parse a comma-separated size string into a tuple of integers.

    Parameters
    ----------
    value
        A comma-separated string of three integers, e.g., "10,1440,721"

    Returns
    -------
    tuple[int, int, int]
        A tuple of (time, latitude, longitude) dimensions.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value cannot be parsed as three integers.
    """
    try:
        parts = value.split(",")
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid size format '{value}'. Expected 'time,latitude,longitude' "
            f"(e.g., '10,1440,721'): {e}"
        ) from None
    if len(parts) != 3:
        raise ValueError("Expected exactly 3 comma-separated values")
    parts_parsed = [int(p.strip()) for p in parts]
    time, lat, lon = parts_parsed
    return time, lat, lon


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
        prog="geo_regridding",
        description="Benchmark for geo-regridding / reprojection operations",
    )

    parser.add_argument(
        "-s",
        "--shape",
        type=parse_shape,
        default=(365, 1440, 721),
        help=("Input array shape as 'time,latitude,longitude'. Default: '10,1440,721'"),
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
        default="input",
        help="Group name inside the zarr store for input data. Default: 'input'",
    )

    parser.add_argument(
        "--output-group",
        type=str,
        default="output",
        help="Group name inside the zarr store for output data. Default: 'output'",
    )

    parsed = parser.parse_args(args)

    # Default output-path to input-path if not specified
    if parsed.output_path is None:
        parsed.output_path = parsed.input_path

    return parsed


def generate_latitude(size: int) -> np.ndarray:
    """Generate a latitude array."""
    return np.linspace(-90, 90, size)


def generate_longitude(size: int) -> np.ndarray:
    """Generate a longitude array."""
    return np.linspace(-180, 180, size)


def generate_time(size: int) -> np.ndarray:
    """Generate a time array."""
    return np.arange(size)


def ensure_array(
    group: zarr.Group,
    name: str,
    shape: tuple[int, ...],
    generator: Callable[[], np.ndarray],
) -> zarr.Array:
    """Ensure that an array exists and matches the shape and dtype."""
    if name not in group:
        group.create_array(name, data=generator(), chunks=shape)
    obj = group[name]
    if not isinstance(obj, zarr.Array):
        raise TypeError(f"Object {name} is not an array")
    elif obj.shape != shape:
        raise ValueError(f"Array shape mismatch: {obj.shape} != {shape}")
    return obj


def generate_temperature_single(
    array: zarr.Array, shape: tuple[int, ...], seed: int, time_index: int
) -> None:
    """
    Generate a temperature array.

    Parameters
    ----------
    array
        The array to write the data to.
    shape
        The shape of the data to generate.
    seed
        The seed for the random number generator.
    time_index
        The time index to write the data to.

    """
    rng = np.random.RandomState(seed=seed)
    data = rng.standard_normal(shape).astype(np.float32)
    array[time_index, ...] = data


def generate_data(
    input_path: Path,
    input_group: str,
    shape: tuple[int, ...],
) -> None:
    """
    Generate the input zarr data.

    Creates a zarr store with a 'temperature' variable containing random
    float32 data. The array is chunked with time=1 (each time slice in its
    own chunk) and full spatial dimensions.

    Parameters
    ----------
    input_path
        Path to the zarr store root.
    input_group
        Group name for the input data.
    shape
        Tuple of (time, latitude, longitude) dimensions.
    """
    chunks = (1, *shape[1:])

    print(f"Generating data with shape {shape} and chunks {chunks}")
    print(f"Writing to {input_path}/{input_group}/temperature")

    # Open or create the zarr store
    store = zarr.open_group(input_path, mode="a")

    # Create the input group if it doesn't exist
    if input_group not in store:
        group = store.create_group(input_group)
    else:
        group = store[input_group]

    assert isinstance(group, zarr.Group)

    time_dim, lon_dim, lat_dim = shape[:3]

    ensure_array(group, "latitude", (lat_dim,), lambda: generate_latitude(lat_dim))
    ensure_array(group, "longitude", (lon_dim,), lambda: generate_longitude(lon_dim))
    ensure_array(group, "time", (time_dim,), lambda: generate_time(time_dim))

    # Generate temperature in parallel, one per time slice
    if "temperature" not in group:
        temperature = group.create_array(
            "temperature",
            chunks=chunks,
            shape=(time_dim, lat_dim, lon_dim, *shape[3:]),
            dtype=np.float32,
            # overwrite=True,
        )
        # Now fill in parallel
        pool = concurrent.futures.ThreadPoolExecutor()
        futures = []
        for i, time_index in enumerate(range(time_dim)):
            futures.append(
                pool.submit(
                    generate_temperature_single,
                    temperature,
                    shape[1:],
                    i,
                    time_index,
                )
            )

        # raise any errors
        for future in concurrent.futures.as_completed(futures):
            future.result()

    temperature = group["temperature"]
    assert isinstance(temperature, zarr.Array)
    print(
        f"Generated temperature array: shape={shape}, chunks={chunks}, nbytes={temperature.nbytes}"
    )


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


if __name__ == "__main__":
    main()
