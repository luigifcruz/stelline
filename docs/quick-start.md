---
title: Quick Start
description: Load a plugin and run your first flowgraph.
order: 3
category: Getting Started
---

This page assumes you already followed the [installation](/docs/installation) steps and have a working `stelline.cep` bundle. The walkthrough uses an NVIDIA DGX Spark as the host machine, set up following the official [DGX Spark documentation](https://docs.nvidia.com/dgx/dgx-spark/), and the central Stelline plugin, though the same flow applies to any other plugin of the stack. If you are new to CyberEther itself, read the upstream [quick start](https://cyberether.org/docs/quick-start) first to get familiar with the flowgraph editor.

## 1. Load the plugin

Open CyberEther and load `stelline.cep` through the plugin manager. The Stelline blocks will appear in the block picker under the **Stelline** domain. The upstream [installing plugins](https://cyberether.org/docs/installing-plugins) page describes this flow in detail.

## 2. Open an example flowgraph

The plugin bundle ships with example flowgraphs:

- `ata-spectrogram.yml` displays a live spectrum from packet ingest.
- `ata-beamformer.yml` runs a beamforming pipeline.
- `ata-correlator.yml` runs a correlator with UVH5 output.
## 3. Run it

Start the flowgraph and watch the block metrics. The [ATA Receiver](/docs/block-ata-receiver) exposes throughput, packet loss, and queue depths directly on the node, so a healthy ingest is visible at a glance.

## Next steps

- Read the per-block pages in the Stelline category for configuration details, starting with the [ATA Receiver](/docs/block-ata-receiver).
- Read the [Nexus Bridge](/docs/block-nexus-bridge) page to understand how metadata reaches the writer blocks.
