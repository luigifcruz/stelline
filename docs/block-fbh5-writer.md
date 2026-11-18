---
title: FBH5 Writer
description: Sink that records filterbank tensors to FBH5.
order: 43
category: Stelline
---

The FBH5 Writer is the recording sink of a beamformer or spectrometer pipeline. It appends power spectra from GPU memory to an FBH5 file, the HDF5 flavor of the SIGPROC filterbank format used by blimpy and the Breakthrough Listen toolchain. Writes go through the GPUDirect Storage HDF5 driver with the input buffer registered against CUfile, so spectra travel from GPU memory to NVMe without a host bounce.

## How it works

The block is idle until `recording` is enabled. At that point it builds the filterbank header from the input tensor shape and the F-engine band and clock in the flowgraph environment, opens the file through the GPUDirect Storage driver, and allocates the mask dataset that accompanies the data. Each compute cycle then appends one input tensor to the data dataset directly from the registered GPU buffer. The input shape is locked when recording starts, and any change in shape, data type, or contiguity afterwards aborts the write with an error.

The input must carry RF `timestamp` and `duration` attributes. `fch1` and `foff` come from the instance band, with the fine channels per F-engine channel inferred from the input channel count. `tstart` and `tsamp` come from the first accepted block: its first RF sample as a UTC MJD, and its duration divided by the spectra per block, which must divide evenly, times the F-engine sample period. `/data` also carries the exact integer grid as `rf_start_sample` (the F-engine sample count of the first spectrum), `synctime`, and `rf_samples_per_row`. The source name, coordinates, and telescope come from the environment when recording starts: `src_raj` in hours, `src_dej`, `az_start`, and `za_start` in degrees, and `telescope_id` 9, blimpy's ATA identifier, for ATA (0 for any other observatory name). Stock SIGPROC names identifier 9 `140ft`, so a `.fil` export needs an ATA-aware reader to show the telescope correctly. `az_start` and `za_start` are `-1` when the pointing has no azimuth or elevation.

The first written block establishes the sample grid. If a later block reveals a gap, the writer inserts zeroed spectra and sets the corresponding `/mask` entries to `1`. No padding is added before the first block or after the last block. Use a reader that honors `/mask`, because the zeros are not measurements. A change to the F-engine band or clock, a backwards timestamp, changed duration, off-grid timestamp, or gap larger than 1,024 blocks stops the recording before anything is appended.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | string | `./file.fbh5` | Path of the output file. The parent directory must exist. |
| `overwrite` | boolean | `false` | Replace an existing file instead of refusing to start. The existing file is kept until the first block of the new recording is written. |
| `recording` | boolean | `false` | Master switch for writing. |

## Input

| Name | Description |
|---|---|
| `input` | `F32` tensor with time, beams, channels, and intermediate frequency axes, plus `timestamp` and `duration` attributes. |

The time dimension sets how many spectra are appended per compute cycle, and the channel dimension sets the number of frequency channels per spectrum.

## Attributes

| Attribute | Type | Direction | Description |
|---|---|---|---|
| `timestamp` | `U64` | Read from `input` | Required. First sample of the block on the F-engine sample counter. |
| `duration` | `U64` | Read from `input` | Required, positive. Number of F-engine samples in the block. Must divide evenly by the time dimension, and sets `tsamp` and the gap padding grid. |

## Environment variables

The block is a pure consumer of the flowgraph environment. It reads the keys below, where `<name>` is the first antenna of the observation, `<tuning>` is the instance tuning, `<slot>` is a position in the instance band list, and `<band>` is one of the instance band indices.

Read when recording starts and checked again before every write. The recording stops if any of them goes missing or becomes invalid, or if the first antenna, the tuning, the set of bands, the merged band range, or the clock changes:

| Key | Purpose |
|---|---|
| `observation.antennas.0` | Antenna whose F-engine band, clock, and pointing define the file. |
| `instance.bands.len` | Band count of this instance, at least one. Multiple bands must share one tuning, compared case-insensitively, and form one contiguous channel and frequency range with a uniform channel width. Repeated bands are ignored, and the writer records the bands as a single band. |
| `instance.bands.<slot>.tuning` | Tuning identifier. |
| `instance.bands.<slot>.band_index` | Band index within the tuning. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.frequency_start` | Band start frequency in MHz. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.frequency_stop` | Band stop frequency in MHz. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.channel_start` | First coarse channel. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.channel_stop` | Last coarse channel, exclusive. |
| `observatory.antenna.<name>.tunings.<tuning>.fengine.synctime` | F-engine sync time, the timestamp origin. |
| `observatory.antenna.<name>.tunings.<tuning>.fengine.sample_period` | F-engine sample period in seconds. |

Read once when recording starts. Azimuth and elevation are optional:

| Key | Purpose |
|---|---|
| `observatory.name` | Selects `telescope_id`. |
| `observatory.antenna.<name>.pointing.source_name` | `source_name`. |
| `observatory.antenna.<name>.pointing.ra` | Right ascension in radians, written as `src_raj` in hours. |
| `observatory.antenna.<name>.pointing.dec` | Declination in radians, written as `src_dej` in degrees. |
| `observatory.antenna.<name>.pointing.az` | Azimuth in radians, written as `az_start` in degrees. |
| `observatory.antenna.<name>.pointing.el` | Elevation in radians, written as `za_start`, the zenith angle in degrees. |

In a live deployment all of these come from the [Nexus Bridge](/docs/block-nexus-bridge).

## Metrics

The node reports the total data written, the current write bandwidth in MB/s, and the number of chunks committed to the file.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `bandwidth` | Write bandwidth in megabytes per second. |
| `totalDataWritten` | Total written FBH5 payload in megabytes. |
| `chunksWritten` | Total number of written FBH5 chunks, including padded chunks inserted for RF gaps. |

## Combining bands

Instances recording adjacent bands write one file each, on a shared RF grid but from different start times. `tools/stitcher.py` joins them into one file covering their common time range:

```shell
$ python3 tools/stitcher.py combined.fbh5 band0.fbh5 band1.fbh5 band2.fbh5
```

The inputs can be given in any order. The tool refuses files with a different `synctime`, `rf_samples_per_row`, channel width, or IF count, rows that are off a shared grid, and bands with a gap or overlap between them. It trims each input to the latest start and earliest end using `rf_start_sample`, joins `/data` and `/mask` along frequency, and keeps the header of the band holding the first output channel, with `nchans`, `tstart`, and `rf_start_sample` updated. An existing output is kept unless `--overwrite` is given. Convert the result with blimpy's `h52fil` for tools that read SIGPROC `.fil` files. That format has no mask.

## Requirements

The GPUDirect Storage stack must be installed. See [host dependencies](/docs/host-dependencies) for the setup.
