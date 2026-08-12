---
title: UVH5 Reader
description: Source that reads a UVH5 file.
order: 46
category: Stelline
---

The UVH5 reader is the source of an offline correlation/visibilities post-processing pipeline. It reads correlated integrations from a UVH5 file. When CUDA is set as the device, reads come via the GPUDirect Storage HDF5 driver, so data travel to GPU memory from NVMe without a host bounce.

## How it works

The block is idle until `playing` is enabled. At that point it starts iterating through the `/Data/visdata` in chunks of `batchSize`*`Nbls`, where `batchSize` effectively selects how many time indices to chunk per step. With the block's device set as CUDA all of this happens via GPUDirect Storage driver. The `batchSize` must be an integer factor of the T dimension in the data. Each compute cycle then steps through the data until it reaches the end at which point it will loop back to the beginning. If `loop` is enabled `playing` will be left enabled and the dataset will repeat, otherwise `playing` will be disabled.

 Currently the `/Data/flags` and `/Data/nsamples` datasets are not read from the file.

The observational header fields (source name, coordinates, channel frequencies etc) are pushed to the flowgraph environment under the `observatory` key. This does need revision however and is just a mostly good enough first pass implementation.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | string | `./file.uvh5` | Path of the input file. |
| `batchSize` | int | `1` | The step-size for the `Time` dimension of the incremental read. |
| `playing` | boolean | `false` | Master switch for reading. |
| `loop` | boolean | `false` | Loop back to the start after reaching the end of the file. |

## Output

| Name | Description |
|---|---|
| `signal` | Contiguous `{CF32|CF64}` tensor shaped `[times, baselines, channels, polarisation-product]` read from the file. |

The `times` dimension is set by the `batchSize` configuration value, everything else is determined by the contents of the file being read.

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