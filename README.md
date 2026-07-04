# Stelline

Radio astronomy blocks for [CyberEther](https://cyberether.org).

Stelline is a CyberEther plugin that ingests telescope data straight off the network into GPU memory and writes it out to disk in standard radio astronomy formats. It currently targets the [Allen Telescope Array](https://www.seti.org/ata), using RDMA packet capture on the receive side and GPUDirect Storage on the write side, so data can flow from the NIC through GPU compute to NVMe without bouncing through host memory.

Stelline is part of the larger [stelline.space](https://stelline.space) stack for real-time sky observation.

## Blocks

| Block | Kind | Description |
|---|---|---|
| `ata_receiver` | Network | High-throughput UDP/multicast ingest of ATA voltage packets directly into GPU buffers. |
| `uvh5_writer` | Storage | Writes correlated visibilities to [UVH5](https://pyuvdata.readthedocs.io/en/latest/uvh5_format.html) (HDF5) using GPUDirect Storage. |
| `fbh5_writer` | Storage | Writes filterbank spectra to FBH5 (HDF5) using GPUDirect Storage. |

These compose with CyberEther's built-in DSP and visualization modules (FFT, casting, spectrogram, lineplot, etc.) inside a flowgraph.

## Example Flowgraphs

The `examples/` directory contains complete pipelines that are also bundled into the plugin package:

- Live spectrum display from packet ingest (`ata-spectrogram.yml`).
- Beamforming pipeline (`ata-beamformer.yml`).
- Correlator with UVH5 output (`ata-correlator.yml`).

## Requirements

- Linux with an NVIDIA GPU and a recent CUDA toolkit.
- [CyberEther](https://github.com/luigifcruz/CyberEther) (Jetstream) 1.5.0 or newer.
- An NVIDIA ConnectX NIC with Mellanox OFED for the networking blocks.
- The GPUDirect Storage stack (`nvidia-fs`) for the HDF5 writer blocks.

Host setup is involved (driver, OFED, GDS, hugepages), so the provided Docker environment is the recommended way to get started.

## Building

Stelline is a standard Meson project:

```
meson setup build
meson compile -C build
```

This produces `build/stelline.cep`, the CyberEther plugin bundle (plugin library, manifest, and example flowgraphs), ready to be loaded by CyberEther's plugin manager.

## Development Environment

The provided Docker images ship a ready-to-use development environment with all dependencies preinstalled, exposed through JupyterLab or code-server:

```
docker build -t stelline-base -f docker/Dockerfile-base .
docker build -t stelline -f docker/Dockerfile-dev .
```

## License

Stelline is distributed under the MIT License. See [LICENSE](LICENSE).
