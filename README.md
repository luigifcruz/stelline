# Stelline

A GPU-accelerated radio astronomy plugin for [CyberEther](https://cyberether.org).

Stelline is a CyberEther plugin from the [stelline.space](https://stelline.space) stack. It captures telescope voltage packets straight off the network into GPU memory with RDMA, processes them with the DSP blocks formerly provided by BLADE, and writes finished data products to disk in standard radio astronomy formats with GPUDirect Storage. The GPU data path runs from NIC through compute to NVMe without bouncing through host memory. The blocks currently target the [Allen Telescope Array](https://www.seti.org/ata).

The DSP blocks provide channelization, beamforming, correlation, phasor generation, polarization conversion, detection, integration, and stacking. The plugin includes CPU implementations alongside CUDA implementations. The Phasor runs on CPU. Building Stelline still requires CUDA.

## Documentation

The full documentation lives at [stelline.space/docs](https://stelline.space/docs). It covers [installation](https://stelline.space/docs/installation), host and system setup, and reference pages for the network, storage, and DSP blocks.

## Example Flowgraphs

The `examples/` directory contains complete pipelines that are also bundled into the plugin package:

- Live spectrum display from packet ingest (`ata-spectrogram.yml`).
- Beamforming pipeline (`ata-beamformer.yml`).
- Correlator with UVH5 output (`ata-correlator.yml`).

## Building

Build instructions, container images, and host setup are covered in the [installation documentation](https://stelline.space/docs/installation).

## License

Stelline is distributed under the MIT License. See [LICENSE](LICENSE).
