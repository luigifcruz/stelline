---
title: Overview
description: A GPU-accelerated radio astronomy stack built on CyberEther.
order: 1
category: Getting Started
---

Stelline is a stack for GPU-accelerated radio astronomy. It builds on [CyberEther](https://cyberether.org) flowgraphs to run complete telescope backends, from voltage packets arriving at the network interface, through machine learning and data analysis, to writing standard radio astronomy products to disk. The stack currently targets the [Allen Telescope Array](https://www.seti.org/ata), but it is built to be extended to other telescopes, with more coming soon.

## Plugins

The stack is delivered as CyberEther plugins, each contributing its own set of blocks to the flowgraph editor.

### Stelline

The Stelline plugin captures voltage packets from the network directly into GPU memory, processes them with the DSP blocks formerly delivered by BLADE, writes finished data products to disk in standard radio astronomy formats, and mirrors the Nexus metadata service into the flowgraph so recordings stay fully described. Both ends of the GPU data path are hardware offloaded, with RDMA on the capture side and GPUDirect Storage on the write side. These DSP blocks also include CPU implementations, with the Phasor running on CPU. Its blocks combine freely with CyberEther's [built-in blocks](https://cyberether.org/docs/blocks).

| Block | Kind | Description |
|---|---|---|
| [ATA Receiver](/docs/block-ata-receiver) | Network | High-throughput ingest of ATA voltage packets directly into GPU buffers. |
| [UVH5 Writer](/docs/block-uvh5-writer) | Storage | Writes correlated visibilities to UVH5 (HDF5) using GPUDirect Storage. |
| [FBH5 Writer](/docs/block-fbh5-writer) | Storage | Writes filterbank spectra to FBH5 (HDF5) using GPUDirect Storage. |
| [Nexus Bridge](/docs/block-nexus-bridge) | Network | Streams Nexus metadata into the flowgraph environment. |
| [Channelizer](/docs/block-channelizer) | DSP | Splits coarse-frequency voltages into centered fine-frequency channels. |
| [Beamformer](/docs/block-beamformer) | DSP | Forms coherent and optional incoherent beams from antenna voltages and phasors. |
| [Phasor](/docs/block-phasor) | DSP | Generates calibrated beamforming weights from telescope metadata. |
| [Correlator](/docs/block-correlator) | DSP | Cross-correlates antenna voltages into baseline visibilities. |
| [Polarizer](/docs/block-polarizer) | DSP | Converts or extracts voltage polarization components. |
| [Detector](/docs/block-detector) | DSP | Detects and integrates voltage power products. |
| [Integrator](/docs/block-integrator) | DSP | Sums samples within or across input buffers. |
| [Stacker](/docs/block-stacker) | DSP | Tiles successive input buffers along a tensor axis. |

### Migrating from BLADE

The BLADE plugin is being retired, and its eight DSP blocks are now part of Stelline. Their block identifiers, configuration fields, and tensor interfaces are unchanged, so existing flowgraphs can use the Stelline bundle without renaming blocks. Remove the BLADE bundle before loading Stelline to avoid duplicate block registrations. For C++ integrations, replace `<blade/...>` includes with `<stelline/...>`. The `Jetstream::Blocks` and `Jetstream::Modules` types are unchanged.

## Requirements

Stelline requires [CyberEther](https://github.com/luigifcruz/CyberEther). Beyond that, requirements differ between blocks, so check the Requirements section of each block page. The [System Setup](/docs/considerations) category covers the host side in detail.

Check out the [installation](/docs/installation) page to get started!
