---
title: Nexus Bridge
description: Streams Nexus metadata into the flowgraph environment.
order: 44
category: Stelline
---

The Nexus Bridge connects a flowgraph to Nexus, the metadata service of the Stelline stack. It keeps the flowgraph environment synchronized with live observatory state, so recording sinks like the [UVH5 Writer](/docs/block-uvh5-writer) always see current antenna positions, pointing, and Earth orientation data without any block-to-block wiring.

## How it works

On creation the block starts a background watcher thread that reads metadata from the Nexus telemetry service. Every five seconds it posts `{"instanceId": NEXUS_INSTANCE_ID}` to `POST /v1/otl/metadata`, and Nexus returns the current observatory snapshot filtered and completed for that instance and its observation. The watcher computes the delta against the previous snapshot and places it on a queue. If authentication fails or the service is unreachable, the watcher marks the bridge disconnected and retries on an interval.

The block itself is throttled, so it wakes periodically rather than spinning. Each cycle it drains the queued deltas and applies them to the flowgraph environment: changed entries are written under their original Nexus key, and entries that disappeared from Nexus are removed. Every mirrored entry keeps the Nexus value, type tag (such as `string`, `u64`, or `f64`), and optional validity interval. Integer values are represented as decimal strings in the flowgraph environment. The compute path never blocks on the network, since all communication happens on the watcher thread.

The bridge also samples the block metrics exposed through CyberEther's Python runtime every five seconds. It selects descriptors with type `stelline-metrics-number` or `stelline-metrics-string`. These metrics use `visibility: internal` to stay hidden in the node while remaining available to Python. It sends non-empty snapshots to `POST /v1/instances/<NEXUS_INSTANCE_ID>/metrics` on a separate publisher thread, so telemetry uploads never block flowgraph computation. Each snapshot is a sample with its own `eventId`. A network failure, `429`, or `5xx` keeps the sample and retries it with the same ID, so Nexus never stores it twice. Any other rejection drops it. Up to 256 unsent samples are kept, and the oldest are discarded first. Samples are sent in batches of at most 16 and 768 KiB, below Nexus's 1 MiB request limit, and a snapshot whose metrics exceed Nexus's 512 KiB per-sample limit is skipped.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | string | `https://nexus.stelline.space` | Nexus deployment. Only the scheme and host are used, and the telemetry service is reached at `<url>/telemetry`. `NEXUS_TELEMETRY_URL` takes precedence when set. |

Changing the URL at runtime recreates the block and restarts the watcher.

## Environment variables

The bridge reads these process environment variables:

| Variable | Description |
|---|---|
| `NEXUS_INSTANCE_ID` | Instance used to request contextual metadata and publish metrics. Required; Nexus-managed Docker workloads set this automatically. |
| `NEXUS_TELEMETRY_URL` | Base URL of the Nexus telemetry service. The Replicant sets it at container launch. When unset, it is derived from the block's `url`. |
| `NEXUS_INSTANCE_CREDENTIALS_FILE` | Path to a file holding the instance bearer JWT. The Replicant mounts it read-only. Preferred over `NEXUS_INSTANCE_CREDENTIALS`. |
| `NEXUS_INSTANCE_CREDENTIALS` | Bearer JWT, without the `Bearer ` prefix. Used when the file is not set. Nexus issues it for this instance through the observation's scheduled end plus ten minutes. |
| `NEXUS_OTL_WATCH_RETRY_SECONDS` | Delay between metadata retries after a failure, in seconds. Defaults to `5`. |

The bridge sends the token as `Authorization: Bearer <token>` on every request. A missing token appears in `last_error`. The token is not copied into the flowgraph environment. Nexus-managed multi-day observations require no token renewal: the bearer lasts for the scheduled observation, and Nexus rejects it when the instance stops or the observation is cancelled or completed. Because the bridge reads the credentials file, the Replicant can run with `NEXUS_WORKLOAD_SECRET_ENV=false`, which keeps the bearer out of `docker inspect`.

The block mirrors every metadata key published by Nexus verbatim, including the `observatory.*`, `observation.*`, and `instance.*` families consumed by the writer blocks, and additionally publishes its own status.

Published flowgraph environment keys:

| Key | Description |
|---|---|
| `nexus.bridge` | Bridge status with `connected`, `variables_loaded`, `metrics_monitored`, `url` (the telemetry service), and `last_error` fields. |

## Metrics

The node reports **Connected**, which flips once the first snapshot arrives, **Variables Loaded**, the number of entries currently mirrored, and **Metrics Monitored**, the number of block metrics accepted into Nexus telemetry snapshots. These values are also readable by other blocks through the `nexus.bridge` status key, and `last_error` in that key carries the reason for the most recent disconnect.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `connected` | Whether the bridge has received a Nexus metadata snapshot. |
| `variablesLoaded` | Number of Nexus metadata variables currently mirrored into the flowgraph environment. |
| `metricsMonitored` | Number of block metrics currently selected for Nexus telemetry. |

## Requirements

The block uses only the Python standard library, so it needs no extra packages in the selected Python runtime.
