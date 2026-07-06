# Stelline

Radio astronomy blocks for [CyberEther](https://cyberether.org).

Stelline is a CyberEther plugin that ingests telescope data straight off the network into GPU memory and writes it out to disk in standard radio astronomy formats. It currently targets the [Allen Telescope Array](https://www.seti.org/ata), using RDMA packet capture on the receive side and GPUDirect Storage on the write side, so data can flow from the NIC through GPU compute to NVMe without bouncing through host memory.

Stelline is part of the larger [stelline.space](https://stelline.space) stack for GPU-accelerated radio astronomy.

## Documentation

The full documentation lives at [stelline.space/docs](https://stelline.space/docs). It covers [installation](https://stelline.space/docs/installation), host and system setup, and a reference page for every block: the ATA Receiver, the UVH5 Writer, the FBH5 Writer, and the Nexus Bridge.

## Example Flowgraphs

The `examples/` directory contains complete pipelines that are also bundled into the plugin package:

- Live spectrum display from packet ingest (`ata-spectrogram.yml`).
- Beamforming pipeline (`ata-beamformer.yml`).
- Correlator with UVH5 output (`ata-correlator.yml`).

## Building

Build instructions, container images, and host setup are covered in the [installation documentation](https://stelline.space/docs/installation).

## License

Stelline is distributed under the MIT License. See [LICENSE](LICENSE).
