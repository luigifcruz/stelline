# Stelline

Stelline is a radio-astronomy plugin for [CyberEther](https://cyberether.org). It extends CyberEther's GPU-accelerated signal-processing and visualization environment with telescope networking, observatory I/O, and radio-astronomy workflow components.

The project is part of the broader Stelline stack for real-time sky observations. More context is available at [stelline.space](https://stelline.space).

## Overview

CyberEther provides the flowgraph interface, heterogeneous GPU execution layer, and real-time visualization tools. Stelline adds astronomy-specific modules for working with telescope data streams, including Allen Telescope Array workflows, high-throughput packet ingest, and HDF5-based storage paths.

Stelline is designed for systems that need to move radio data directly from network interfaces through GPU compute and into visualization or storage pipelines with minimal overhead.

## What Is Included

- CyberEther plugin library: `libstelline.so`
- ATA-oriented receiver and networking components
- HDF5 writer modules using GPUDirect Storage/VFD-GDS paths
- Example CyberEther flowgraphs in `examples/`
- Docker files for development and deployment environments
- Quickstart and hardware-configuration notes in `docs/`

## Example Flowgraphs

The `examples/` directory contains starter CyberEther graphs for radio-astronomy use cases:

- `examples/ata-correlator.yml`
- `examples/ata-beamformer.yml`
- `examples/ata-spectrogram.yml`

These are intended as reference pipelines for ingesting and processing Allen Telescope Array data inside CyberEther.

## Documentation

Start with the quickstart documentation:

- `docs/quickstart/installation.md`
- `docs/quickstart/configuration.md`
- `docs/quickstart/dependencies.md`

The recommended development path is the provided Docker environment, since Stelline targets GPU-enabled systems with CUDA, CyberEther, Holoscan-related dependencies, GPUDirect Storage, and high-performance networking components.

## License

Stelline is distributed under the MIT License. See `LICENSE`.
