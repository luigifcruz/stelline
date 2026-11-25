## Overview

A radio telescope's digital backend is mostly a problem of moving data fast enough. The digitized antenna voltages arrive as network packets, and copying each packet through the host CPU on the way to a GPU, and again on the way to disk, quickly becomes the bottleneck.[^stelline-repo]

Stelline removes those copies. It captures voltage packets from the network card directly into GPU memory with RDMA, processes them with GPU-accelerated DSP blocks, and writes finished data products to NVMe drives with GPUDirect Storage, which moves data between GPU memory and storage without a bounce buffer in host memory.[^nvidia-gds] It also mirrors the telescope's metadata service into the flowgraph, so recordings carry the observation's metadata. The blocks currently target the Allen Telescope Array.[^welch-ata]
## Capture And Metadata

| Block | What it does |
|---|---|
| ATA Receiver | Captures ATA voltage packets from a ConnectX network card and assembles them into time-aligned GPU tensors. |
| Nexus Bridge | Mirrors Nexus metadata into the flowgraph environment and reports Stelline block metrics back to Nexus. |

## Signal Processing

| Block | What it does |
|---|---|
| Channelizer | Splits each coarse frequency channel into centered fine frequency channels with an FFT. |
| Correlator | Cross-correlates dual-polarization voltages into XX, XY, YX, and YY visibilities for every baseline. |
| Phasor | Computes per-beam geometric delays and calibrated phasor weights from antenna positions and pointing. |
| Beamformer | Forms coherent beams by weighting each antenna with its phasor and summing, plus an incoherent beam. |
| Polarizer | Converts dual linear polarization to a circular basis or extracts a single linear component. |
| Detector | Converts complex voltages into detected power products and integrates them over time. |
| Integrator | Sums samples along an axis to raise the signal-to-noise ratio or reduce the data rate. |
| Stacker | Tiles successive buffers along an axis into the larger geometry writers and displays expect. |

## Recording And Playback

| Block | What it does |
|---|---|
| UVH5 Writer | Writes correlated visibilities to UVH5, the HDF5 format for interferometric data,[^pyuvdata] with GPUDirect Storage. |
| UVH5 Reader | Reads a UVH5 file back into the flowgraph for offline processing. |
| FBH5 Writer | Writes filterbank spectra to FBH5, the HDF5 layout of Breakthrough Listen's archive,[^lebofsky-bl] with GPUDirect Storage. |
| FBH5 Reader | Reads power spectra and their mask from an FBH5 file back into the flowgraph. |

## Requirements

Stelline needs a Linux CUDA build of CyberEther 2.0.0 or newer. Live capture needs an NVIDIA ConnectX network card, the writers need the GPUDirect Storage stack, and the Nexus Bridge needs credentials for the Nexus service. The DSP blocks also have CPU implementations, so offline pipelines that start from a UVH5 or FBH5 file run without that hardware.

## Documentation

The [Stelline documentation](https://stelline.space/docs) has a reference page for each block and covers [installation](https://stelline.space/docs/installation) and host setup. The [host dependencies](https://stelline.space/docs/host-dependencies) and [system configuration](https://stelline.space/docs/system-configuration) pages walk through preparing a capture machine.

Every Stelline block ID starts with `stelline_`. Flowgraphs written for the retired BLADE plugin need their `module` entries renamed, for example `beamformer` to `stelline_beamformer`.

## References

[^welch-ata]: J. Welch *et al.*, "The Allen Telescope Array: The first widefield, panchromatic, snapshot radio camera for radio astronomy and SETI," *Proc. IEEE*, vol. 97, no. 8, pp. 1438-1447, Aug. 2009, [doi:10.1109/JPROC.2009.2017103](https://doi.org/10.1109/JPROC.2009.2017103), [arXiv:0904.0762](https://arxiv.org/abs/0904.0762).
[^stelline-repo]: L. Cruz *et al.*, *Stelline*, GitHub. [github.com/luigifcruz/stelline](https://github.com/luigifcruz/stelline)
[^nvidia-gds]: NVIDIA, "[GPUDirect Storage overview guide](https://docs.nvidia.com/gpudirect-storage/overview-guide/index.html)," NVIDIA Documentation.
[^pyuvdata]: G. K. Keating *et al.*, "pyuvdata v3: An interface for astronomical interferometric data sets in Python," *J. Open Source Softw.*, vol. 10, no. 109, art. no. 7482, May 2025, [doi:10.21105/joss.07482](https://doi.org/10.21105/joss.07482).
[^lebofsky-bl]: M. Lebofsky *et al.*, "The Breakthrough Listen search for intelligent life: Public data, formats, reduction, and archiving," *Publ. Astron. Soc. Pac.*, vol. 131, no. 1006, art. no. 124505, Dec. 2019, [doi:10.1088/1538-3873/ab3e82](https://doi.org/10.1088/1538-3873/ab3e82), [arXiv:1906.07391](https://arxiv.org/abs/1906.07391).
