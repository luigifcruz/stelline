---
title: FBH5 Reader
description: Source that reads the `/data` dataset from an FBH5 file.
order: 45
category: Stelline
---

The FBH5 reader is the source of an offline beamformed-data post-processing pipeline. It reads beamformed and detected (either Stokes I or {XX, XY, YX, YY}) spectra from an FBH5 file, the HDF5 flavor of the SIGPROC filterbank format used by blimpy and the Breakthrough Listen toolchain. When CUDA is set as the device, reads come via the GPUDirect Storage HDF5 driver, so spectra travel to GPU memory from NVMe without a host bounce.

## How it works

The block is idle until `playing` is enabled. At that point it starts iterating through the `/data` in chunks of `batchSize`, producing tensors with shape `{T=batchSize, P=NIFs, F=NFreqs}` where the last 2 dimensions reflect the data shape within the file. The `batchSize` must be an integer factor of the T dimension in the data. The same chunking happens for the `/mask` dataset that accompanies the data. With the block's device set as CUDA all of this happens via GPUDirect Storage driver. Each compute cycle then steps through the data until it reaches the end at which point it will loop back to the beginning. If `loop` is enabled `playing` will be left enabled and the dataset will repeat, otherwise `playing` will be disabled.

The observational header fields (source name, coordinates, channel frequencies etc) are pushed to the flowgraph environment under the `observatory` key.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | string | `./file.fbh5` | Path of the input file. |
| `batchSize` | int | `1` | The step-size for the `Time` dimension of the incremental read. |
| `playing` | boolean | `false` | Master switch for reading. |
| `loop` | boolean | `false` | Loop back to the start after reaching the end of the file. |

## Output

| Name | Description |
|---|---|
| `signal` | Contiguous `{F32|F64}` tensor shaped `[time, polarisation-product, channels]` read from the file. |
| `mask` | Contiguous `U8` boolean tensor shaped `[time, polarisation-product, channels]` read from the file that flags data in the `signal` tensor. |

The time dimension is set by the `batchSize` configuration value, everything else is determined by the contents of the file being read.

## Metrics

The node reports the current position of the read within the file and the current read bandwidth in MB/s.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `progress` | Current file position as a percentage. |
| `currentBandwidth` | Read bandwidth in megabytes per second. |

## Requirements

If using the CUDA device, the GPUDirect Storage stack must be installed. See [host dependencies](/docs/host-dependencies) for the setup.