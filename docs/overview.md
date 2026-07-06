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

The Stelline plugin is the foundation of the stack. It captures voltage packets from the network directly into GPU memory, writes finished data products to disk in standard radio astronomy formats, and mirrors the Nexus metadata service into the flowgraph so recordings stay fully described. Both ends of the data path are hardware offloaded, with RDMA on the capture side and GPUDirect Storage on the write side, keeping the host CPU out of the loop. Its blocks combine freely with CyberEther's [built-in blocks](https://cyberether.org/docs/blocks).

| Block | Kind | Description |
|---|---|---|
| [ATA Receiver](/docs/block-ata-receiver) | Network | High-throughput ingest of ATA voltage packets directly into GPU buffers. |
| [UVH5 Writer](/docs/block-uvh5-writer) | Storage | Writes correlated visibilities to UVH5 (HDF5) using GPUDirect Storage. |
| [FBH5 Writer](/docs/block-fbh5-writer) | Storage | Writes filterbank spectra to FBH5 (HDF5) using GPUDirect Storage. |
| [Nexus Bridge](/docs/block-nexus-bridge) | Network | Streams Nexus metadata into the flowgraph environment. |

### Blade

Blade is the digital signal processing engine of the stack, with GPU beamforming and correlation kernels developed for the Allen Telescope Array. It is delivered as its own plugin, contributing the compute blocks that sit between the receiver and the writers. Its documentation will join this site soon.

## Requirements

Stelline requires [CyberEther](https://github.com/luigifcruz/CyberEther) 1.6.0 or newer. Beyond that, requirements differ between blocks, so check the Requirements section of each block page. The [System Setup](/docs/considerations) category covers the host side in detail.

Check out the [installation](/docs/installation) page to get started!
