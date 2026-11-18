---
title: Stacker
description: Stacks the input along a specified axis.
order: 57
category: Stelline
---

The Stacker tiles successive input buffers along a tensor axis, assembling an output whose axis is `ratio` times larger than the input. It is the standard way to batch short buffers into the larger block geometry expected by writers, correlators, or waterfall displays.

## How it works

The output is zeroed when a new stacking cycle begins. Each compute cycle then writes the current input buffer into its slot along the chosen axis, cycling through `ratio` slots, and the assembled output is only emitted once every slot has been filled. In between, the block skips downstream processing. A ratio of one bypasses stacking. The CPU implementation copies contiguous rows directly. On CUDA, narrow rows use a runtime-compiled kernel and wide rows use a strided device copy.

With valid `U64` `timestamp` and `duration` attributes, multi-block windows start at exact multiples of `duration × ratio` in F-engine samples and require adjacent, equal-duration blocks. A forward gap or a span change breaks an incomplete window; stale and overlapping blocks are ignored. Inputs with neither timing attribute retain count-based grouping; partial or invalid timing is skipped. An emitted tensor carries the window start and full RF span; a single-block result retains its input interval.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `axis` | integer | `0` | The tensor axis to stack along. |
| `ratio` | integer | `1` | Number of buffers tiled into one output. Must be greater than zero. |
| `copySizeThreshold` | integer | `512` | Chunk width on CUDA below which the kernel path is used instead of a strided copy. Ignored on CPU. |
| `blockSize` | integer | `512` | Number of CUDA threads per block for the kernel path. Ignored on CPU. |

## Input

| Name | Description |
|---|---|
| `buffer` | Contiguous `F32`, `CF32`, or `CI8` tensor of any rank greater than the chosen axis. |

## Output

| Name | Description |
|---|---|
| `buffer` | Tensor with the input type, the chosen axis multiplied by `ratio`, and all other axes unchanged. |
