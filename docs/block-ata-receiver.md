---
title: ATA Receiver
description: Receives ATA voltage packets and assembles output tensors.
order: 41
category: Stelline
---

The ATA Receiver is the ingest point of a Stelline pipeline. It captures Allen Telescope Array voltage packets from a ConnectX NIC and reassembles them into dense, time-aligned GPU tensors that downstream correlation, beamforming, or spectral blocks can consume directly. Packet reception is built on the Holoscan Advanced Network operator, so payloads land in GPU memory through kernel bypass without touching the host.

## How it works

The ATA distributes voltage data as UDP multicast streams. Each packet carries a 16-byte application header with the antenna ID, the first channel number, the channel count, and a timestamp in F-engine sample units, followed by a fixed 6144-byte payload. One packet therefore holds a small fragment of the full antenna and channel space, described by the `partialBlock` shape.

The receiver joins the multicast groups listed in `subscriptions` on the configured interface and spreads reception across the worker cores. Incoming packets are validated against the header fields, filtered by `offsetBlock`, and slotted into an in-flight block keyed by their timestamp. Once every fragment of a block has arrived, or the block is forced out by newer data, a CUDA kernel gathers the fragments into a contiguous output tensor with the `totalBlock` shape. Finished tensors are recycled through a fixed-size output pool, so a slow consumer shows up as pool exhaustion instead of unbounded memory growth.

Packets older than the eviction cutoff are dropped and counted, which makes late or out-of-order network delivery visible in the metrics rather than silently corrupting blocks.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `interfaceAddress` | string | | Address of the network interface that joins the multicast groups. |
| `gpuDeviceId` | integer | `0` | CUDA device that receives the payloads and runs the gather kernel. |
| `masterCore` | integer | `0` | CPU core pinned to the receiver control thread. |
| `workerCores` | list of integers | | CPU cores pinned to the packet reception workers. At least one is required. |
| `subscriptions` | string | | Multicast subscriptions, one per line (see below). |
| `totalBlock` | list of integers | `[1, 1, 1024, 1]` | Output tensor shape as `[antennas, channels, samples, polarizations]`. |
| `partialBlock` | list of integers | `[1, 1, 1024, 1]` | Fragment shape carried by a single packet. |
| `offsetBlock` | list of integers | `[0, 0, 0, 0]` | Origin of the captured region within the full telescope output. |
| `dataType` | string | `CF32` | Output tensor data type, either `CF32` or `CI8`. |
| `packetsPerBurst` | integer | | Number of packets in one receive burst. |
| `maxConcurrentBursts` | integer | | Cap on bursts held by the receiver at once. |
| `maxConcurrentBlocks` | integer | `4` | Cap on partially filled blocks kept in flight. |
| `outputPoolSize` | integer | `2` | Number of output tensors cycled through the pool. |

The `interfaceAddress`, `workerCores`, `subscriptions`, `packetsPerBurst`, and `maxConcurrentBursts` parameters have no usable defaults and must be set.

### Subscription syntax

Each line selects one stream and must follow this form:

```
- source_ip:port -> destination_ip:port
```

Either side of a colon accepts `*` as a wildcard, so `- * -> 239.1.2.3:50000` accepts any source sending to that multicast group and port. The destination groups are joined on `interfaceAddress` at startup.

### Shape rules

- All three shape parameters have exactly four dimensions, ordered `[antennas, channels, samples, polarizations]`.
- Each `totalBlock` axis must be a multiple of the matching `partialBlock` axis.
- The sample and polarization entries of `offsetBlock` must be zero.
- The antenna entry of `offsetBlock` must be a multiple of the `partialBlock` antenna count.

## Output

| Name | Description |
|---|---|
| `output` | Tensor shaped as `totalBlock` in the configured data type, carrying a `timestamp` attribute with the block start time in F-engine sample units. |

The `timestamp` attribute is the link between the data path and the metadata path. Sinks such as the [UVH5 Writer](/docs/block-uvh5-writer) use it to look up time-matched pointing and Earth orientation values.

## Attributes

| Attribute | Type | Direction | Description |
|---|---|---|---|
| `timestamp` | `U64` | Written on `output` | Start time of the block in F-engine sample units. |

## Metrics

The node surfaces its runtime health as metrics, the most important being:

- **Throughput** in Gbps, averaged since start.
- **Packets** received, evicted (filtered or late), and lost (no block could be allocated).
- **Blocks** received, computed, and lost.
- **Bursts in flight** against `maxConcurrentBursts`, with the average burst release time.
- **Queue depths** for the idle, receive, and compute stages, plus output pool availability.
- **Observed antennas and channels**, both the full set seen on the wire and the subset accepted after filtering, which is the fastest way to diagnose a wrong `offsetBlock`.

A healthy steady state shows zero lost packets, bursts in flight well below the cap, and the observed antenna and channel sets matching the configured capture window.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `blocksReceived` | Total completed blocks submitted to the gather kernel. |
| `blocksComputed` | Total completed blocks emitted from compute submit. |
| `blocksLost` | Total stale or evicted blocks. |
| `blocksEmitted` | Total blocks successfully output from the module. |
| `packetsReceived` | Total received packets accepted into blocks. |
| `packetsEvicted` | Packets discarded by offset or cutoff filtering. |
| `packetsLost` | Packets dropped because no block could be allocated. |
| `idleQueue` | Current idle queue depth. |
| `receiveQueue` | Current receive queue depth. |
| `computeQueue` | Current compute queue depth. |
| `burstsInFlight` | Current number of in-flight bursts. |
| `avgBurstReleaseTimeUs` | Average burst release time in microseconds. |
| `memPoolAvailable` | Current reusable output tensor pool availability. |
| `memPoolReferenced` | Current reusable output tensor pool references. |
| `blockMapLatestTimeIndex` | Latest known block time index. |
| `blockMapUsed` | Current number of active block map entries. |
| `blockMapCapacity` | Maximum number of concurrent block map entries. |
| `payloadSizes` | Observed payload sizes. |
| `allAntennas` | All observed antenna identifiers. |
| `filteredAntennas` | Accepted antenna identifiers after filtering. |
| `allChannels` | All observed channel identifiers. |
| `filteredChannels` | Accepted channel identifiers after filtering. |
| `latestTimestamp` | Latest accepted packet timestamp. |

## Requirements

Reception requires an NVIDIA ConnectX NIC with Mellanox OFED and hugepages configured on the host. See [host dependencies](/docs/host-dependencies) for the setup and [considerations](/docs/considerations) for the hardware background.
