#!/usr/bin/env python3

import argparse
import contextlib
import os
from pathlib import Path
import sys
import tempfile

import h5py
import numpy as np


GRID_KEYS = ("rf_start_sample", "synctime", "rf_samples_per_row")
BLOCK_BYTES = 64 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MAX_CHANNELS = 2**31 - 1


class StitchError(Exception):
    pass


class Band:
    def __init__(self, path, handle):
        self.path = path
        self.handle = handle
        if "data" not in handle or "mask" not in handle:
            raise StitchError(f"{path}: not an FBH5 file with /data and /mask.")
        self.data = handle["data"]
        self.mask = handle["mask"]
        if self.data.ndim != 3 or self.mask.shape != self.data.shape:
            raise StitchError(f"{path}: /data and /mask must share a [time, ifs, channels] shape.")
        missing = [key for key in GRID_KEYS + ("fch1", "foff", "tstart") if key not in self.data.attrs]
        if missing:
            raise StitchError(f"{path}: missing {', '.join(missing)}, so it was not written with the RF grid fields.")
        self.start = int(self.data.attrs["rf_start_sample"])
        self.synctime = int(self.data.attrs["synctime"])
        self.stride = int(self.data.attrs["rf_samples_per_row"])
        self.fch1 = float(self.data.attrs["fch1"])
        self.foff = float(self.data.attrs["foff"])
        self.rows, self.ifs, self.channels = self.data.shape
        if self.stride <= 0 or self.foff == 0.0:
            raise StitchError(f"{path}: invalid rf_samples_per_row or foff.")

    @property
    def end(self):
        return self.start + self.rows * self.stride


def check_compatible(bands):
    first = bands[0]
    for band in bands[1:]:
        for name, value, expected in (
            ("synctime", band.synctime, first.synctime),
            ("rf_samples_per_row", band.stride, first.stride),
            ("IF count", band.ifs, first.ifs),
            ("data type", band.data.dtype, first.data.dtype),
        ):
            if value != expected:
                raise StitchError(f"{band.path}: {name} {value} differs from {first.path} ({expected}).")
        if abs(band.foff - first.foff) > 1e-9 * abs(first.foff):
            raise StitchError(f"{band.path}: channel width {band.foff} MHz differs from {first.path} ({first.foff} MHz).")
        if (band.start - first.start) % first.stride != 0:
            raise StitchError(f"{band.path}: rows are not on the same RF grid as {first.path}.")


def order_by_frequency(bands):
    ordered = sorted(bands, key=lambda band: band.fch1, reverse=bands[0].foff < 0)
    fch1 = ordered[0].fch1
    foff = ordered[0].foff
    tolerance = 1e-6 * abs(foff)
    column = 0
    for band in ordered:
        for channel in (0, band.channels - 1):
            expected = fch1 + (column + channel) * foff
            actual = band.fch1 + channel * band.foff
            if abs(actual - expected) > tolerance:
                raise StitchError(
                    f"{band.path}: channel {channel} is at {actual:.9f} MHz, off the joined axis at {expected:.9f} MHz. "
                    "the bands have a gap, an overlap, or different channel widths."
                )
        column += band.channels
    return ordered


def chunk_layout(rows, ifs, channels, item_bytes, copy_row_bytes):
    row_bytes = ifs * channels * item_bytes
    if row_bytes <= CHUNK_BYTES:
        depth = max(1, min(rows, CHUNK_BYTES // row_bytes))
        tile = channels
    else:
        depth = max(1, min(rows, BLOCK_BYTES // copy_row_bytes))
        tile = max(1, min(channels, CHUNK_BYTES // (depth * ifs * item_bytes)))
    step = max(depth, (BLOCK_BYTES // copy_row_bytes) // depth * depth)
    return (depth, ifs, tile), step


def copy_attributes(source, target, overrides):
    for name in source.attrs:
        if name in overrides:
            continue
        target.attrs.create(name, source.attrs[name], dtype=source.attrs.get_id(name).dtype)
    for name, value in overrides.items():
        target.attrs.create(name, value, dtype=source.attrs.get_id(name).dtype)


def stitch(inputs, output, overwrite=False):
    output = Path(output)
    resolved = {Path(path).resolve() for path in inputs}
    if len(resolved) != len(inputs):
        raise StitchError("Each input file must be given once.")
    if output.resolve() in resolved or (output.exists() and any(os.path.samefile(output, path) for path in inputs)):
        raise StitchError(f"{output}: the output cannot be one of the inputs.")
    if output.exists() and not overwrite:
        raise StitchError(f"{output}: already exists. Pass --overwrite to replace it.")

    with contextlib.ExitStack() as stack:
        bands = [Band(path, stack.enter_context(h5py.File(path, "r"))) for path in inputs]
        check_compatible(bands)
        bands = order_by_frequency(bands)

        start = max(band.start for band in bands)
        end = min(band.end for band in bands)
        if end <= start:
            raise StitchError("The inputs share no recorded time interval.")
        stride = bands[0].stride
        rows = (end - start) // stride
        channels = sum(band.channels for band in bands)
        if channels > MAX_CHANNELS:
            raise StitchError(f"The joined {channels} channels exceed the FBH5 nchans limit of {MAX_CHANNELS}.")
        ifs = bands[0].ifs
        latest = next(band for band in bands if band.start == start)
        first = bands[0]

        copy_row_bytes = ifs * channels * (first.data.dtype.itemsize + first.mask.dtype.itemsize)
        chunks, step = chunk_layout(rows, ifs, channels, first.data.dtype.itemsize, copy_row_bytes)

        descriptor, staging = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".partial", dir=output.parent)
        try:
            os.close(descriptor)
            with h5py.File(staging, "w") as handle:
                for name in first.handle.attrs:
                    handle.attrs.create(name, first.handle.attrs[name], dtype=first.handle.attrs.get_id(name).dtype)
                data = handle.create_dataset("data", (rows, ifs, channels), dtype=first.data.dtype,
                                             maxshape=(None, ifs, channels), chunks=chunks)
                mask = handle.create_dataset("mask", (rows, ifs, channels), dtype=first.mask.dtype,
                                             maxshape=(None, ifs, channels), chunks=chunks)
                copy_attributes(first.data, data, {
                    "nchans": channels,
                    "tstart": latest.data.attrs["tstart"],
                    "rf_start_sample": np.uint64(start),
                })
                copy_attributes(first.mask, mask, {})

                for row in range(0, rows, step):
                    count = min(step, rows - row)
                    column = 0
                    for band in bands:
                        offset = (start - band.start) // stride + row
                        columns = slice(column, column + band.channels)
                        data[row:row + count, :, columns] = band.data[offset:offset + count]
                        mask[row:row + count, :, columns] = band.mask[offset:offset + count]
                        column += band.channels

            if overwrite:
                os.replace(staging, output)
            else:
                try:
                    os.link(staging, output)
                except FileExistsError:
                    raise StitchError(f"{output}: already exists. Pass --overwrite to replace it.") from None
                os.unlink(staging)
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(staging)
            raise

        return {
            "bands": [band.path for band in bands],
            "rows": rows,
            "channels": channels,
            "rf_start_sample": start,
            "trimmed": {band.path: ((start - band.start) // stride, (band.end - end) // stride) for band in bands},
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Join FBH5 recordings of adjacent bands into one file over their common time range."
    )
    parser.add_argument("output", help="Path of the stitched FBH5 file.")
    parser.add_argument("inputs", nargs="+", help="Adjacent-band FBH5 files, in any order.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the output if it already exists.")
    args = parser.parse_args(argv)

    if len(args.inputs) < 2:
        parser.error("at least two inputs are required.")

    try:
        summary = stitch(args.inputs, args.output, overwrite=args.overwrite)
    except (StitchError, OSError) as error:
        print(f"stitcher: {error}", file=sys.stderr)
        return 1

    print(f"Wrote {args.output}: {summary['rows']} rows x {summary['channels']} channels "
          f"from RF sample {summary['rf_start_sample']}.")
    for path, (head, tail) in summary["trimmed"].items():
        print(f"  {path}: dropped {head} leading and {tail} trailing rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
