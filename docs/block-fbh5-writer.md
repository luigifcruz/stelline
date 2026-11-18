---
title: FBH5 Writer
description: Sink that writes pre-arranged tensors to FBH5.
order: 43
category: Stelline
---

The FBH5 Writer is the recording sink of a beamformer or spectrometer pipeline. It appends power spectra from GPU memory to an FBH5 file, the HDF5 flavor of the SIGPROC filterbank format used by blimpy and the Breakthrough Listen toolchain. Writes go through the GPUDirect Storage HDF5 driver with the input buffer registered against CUfile, so spectra travel from GPU memory to NVMe without a host bounce.

## How it works

The block is idle until `recording` is enabled. At that point it builds the filterbank header from the input tensor shape and the F-engine band and clock in the flowgraph environment, opens the file through the GPUDirect Storage driver, and allocates the mask dataset that accompanies the data. Each compute cycle then appends one input tensor to the data dataset directly from the registered GPU buffer. The input shape is locked when recording starts, and any change in shape, data type, or contiguity afterwards aborts the write with an error.

The input must carry RF `timestamp` and `duration` attributes. `fch1` and `foff` come from the instance band, with the fine channels per F-engine channel inferred from the input channel count. `tstart` and `tsamp` come from the first accepted block: its first RF sample as a UTC MJD, and its duration divided by the spectra per block, which must divide evenly, times the F-engine sample period. `/data` also carries the exact integer grid as `rf_start_sample` (the F-engine sample count of the first spectrum), `synctime`, and `rf_samples_per_row`. The source name and coordinates are still fixed placeholder values.

The first written block establishes the sample grid. If a later block reveals a gap, the writer inserts zeroed spectra and sets the corresponding `/mask` entries to `1`. No padding is added before the first block or after the last block. Use a reader that honors `/mask`; the zeros are not measurements. A change to the F-engine band or clock, a backwards timestamp, changed duration, off-grid timestamp, or gap larger than 1,024 blocks stops the recording before anything is appended.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | string | `./file.fbh5` | Path of the output file. The parent directory must exist. |
| `overwrite` | boolean | `false` | Replace an existing file instead of refusing to start. The existing file is kept until the first block of the new recording is written. |
| `recording` | boolean | `false` | Master switch for writing. |

## Input

| Name | Description |
|---|---|
| `input` | Contiguous `F32` tensor shaped `[time, beams, channels, intermediate frequencies]` with `timestamp` and `duration` attributes. |

The time dimension sets how many spectra are appended per compute cycle, and the channel dimension sets the number of frequency channels per spectrum.

## Attributes

| Attribute | Type | Direction | Description |
|---|---|---|---|
| `timestamp` | `U64` | Read from `input` | Required. First sample of the block on the F-engine sample counter. |
| `duration` | `U64` | Read from `input` | Required, positive. Number of F-engine samples in the block. Must divide evenly by the time dimension, and sets `tsamp` and the gap padding grid. |

## Environment variables

The block is a pure consumer of the flowgraph environment. It reads the keys below when recording starts and checks them again before each write, where `<name>` is the first antenna of the observation, `<tuning>` is the instance tuning, and `<band>` is the instance band index.

| Key | Purpose |
|---|---|
| `observation.antennas.0` | Antenna whose F-engine band and clock define the file axes. |
| `instance.bands.len` | Band count of this instance; must be one. |
| `instance.bands.0.tuning` | Tuning identifier. |
| `instance.bands.0.band_index` | Band index within the tuning. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.frequency_start` | Band start frequency in MHz. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.frequency_stop` | Band stop frequency in MHz. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.channel_start` | First coarse channel. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.channel_stop` | Last coarse channel, exclusive. |
| `observatory.antenna.<name>.tunings.<tuning>.fengine.synctime` | F-engine sync time, the timestamp origin. |
| `observatory.antenna.<name>.tunings.<tuning>.fengine.sample_period` | F-engine sample period in seconds. |

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

## Requirements

The GPUDirect Storage stack must be installed. See [host dependencies](/docs/host-dependencies) for the setup.
