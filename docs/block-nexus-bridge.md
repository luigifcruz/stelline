---
title: Nexus Bridge
description: Streams Nexus metadata into the flowgraph environment.
order: 44
category: Stelline
---

The Nexus Bridge connects a flowgraph to Nexus, the metadata service of the Stelline stack. It keeps the flowgraph environment synchronized with live observatory state, so recording sinks like the [UVH5 Writer](/docs/block-uvh5-writer) always see current antenna positions, pointing, and Earth orientation data without any block-to-block wiring.

## How it works

On creation the block starts a background watcher thread that subscribes to the `observatory:getMetadata` query on the configured Nexus deployment, a Convex application. The subscription is push-based, so the watcher receives a fresh snapshot whenever the observatory state changes, computes the delta against the previous snapshot, and places it on a queue. If the connection drops or the Convex client is unavailable, the watcher marks the bridge disconnected and retries on an interval.

The block itself is throttled, so it wakes periodically rather than spinning. Each cycle it drains the queued deltas and applies them to the flowgraph environment: changed entries are written under their original Nexus key, and entries that disappeared from Nexus are removed. Every mirrored entry keeps the Nexus triple of value, type tag (`text`, `integer`, or `real`), and validity flag. The compute path never blocks on the network, since all communication happens on the watcher thread.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | string | `https://nexus.stelline.space` | Nexus deployment to subscribe to. Only the scheme and host are used. |

Changing the URL at runtime recreates the block and restarts the subscription.

## Environment variables

The block is a pure producer of the flowgraph environment. It mirrors every key published by the Nexus deployment verbatim, including the `observatory.*`, `observation.*`, and `instance.*` families consumed by the writer blocks, and additionally publishes its own status:

| Key | Description |
|---|---|
| `nexus.bridge` | Bridge status with `connected`, `variables_loaded`, `url`, and `last_error` fields. |

## Metrics

The node reports **Connected**, which flips once the first snapshot arrives, and **Variables Loaded**, the number of entries currently mirrored. Both are also readable by other blocks through the `nexus.bridge` status key, and `last_error` in that key carries the reason for the most recent disconnect.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `connected` | Whether the bridge has received a Nexus metadata snapshot. |
| `variablesLoaded` | Number of Nexus metadata variables currently mirrored into the flowgraph environment. |

## Requirements

The block depends on the [Convex](https://pypi.org/project/convex/) Python package to subscribe to the Nexus deployment. Install it on the host with pip:

```bash
python -m pip install convex
```
