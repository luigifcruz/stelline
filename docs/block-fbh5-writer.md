---
title: FBH5 Writer
description: Sink that writes pre-arranged tensors to FBH5.
order: 43
category: Stelline
---

The FBH5 Writer is the recording sink of a beamformer or spectrometer pipeline. It appends power spectra from GPU memory to an FBH5 file, the HDF5 flavor of the SIGPROC filterbank format used by blimpy and the Breakthrough Listen toolchain. Writes go through the GPUDirect Storage HDF5 driver with the input buffer registered against CUfile, so spectra travel from GPU memory to NVMe without a host bounce.

## How it works

The block is idle until `recording` is enabled. At that point it builds the filterbank header from the input tensor shape, opens the file through the GPUDirect Storage driver, and allocates the mask dataset that accompanies the data. Each compute cycle then appends one input tensor to the data dataset directly from the registered GPU buffer. The input shape is locked when recording starts, and any change in shape, data type, or contiguity afterwards aborts the write with an error.

The observational header fields (source name, coordinates, start time, channel frequencies) are currently filled with fixed placeholder values rather than live metadata, so downstream tooling should treat them as provisional until the block is wired to the flowgraph environment like the [UVH5 Writer](/docs/block-uvh5-writer).

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | string | `./file.fbh5` | Path of the output file. The parent directory must exist. |
| `overwrite` | boolean | `false` | Replace an existing file instead of refusing to start. |
| `recording` | boolean | `false` | Master switch for writing. |

## Input

| Name | Description |
|---|---|
| `input` | Contiguous `F32` tensor shaped `[time, beams, channels, intermediate frequencies]`. |

The time dimension sets how many spectra are appended per compute cycle, and the channel dimension sets the number of frequency channels per spectrum.

## Metrics

The node reports the total data written, the current write bandwidth in MB/s, and the number of chunks committed to the file.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `bandwidth` | Write bandwidth in megabytes per second. |
| `totalDataWritten` | Total written FBH5 payload in megabytes. |
| `chunksWritten` | Total number of written FBH5 chunks. |

## Requirements

The GPUDirect Storage stack must be installed. See [host dependencies](/docs/host-dependencies) for the setup.
