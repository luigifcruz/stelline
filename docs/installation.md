---
title: Installation
description: How to install the Stelline plugins.
order: 2
category: Getting Started
---

Stelline plugins ship as CyberEther plugin bundles. The recommended way to install them is through the pre-built bundles, with container images and building from source available as alternatives. This documentation assumes that the host OS is Ubuntu 24.04, but most of the instructions should apply to newer or older versions with minor changes.

## Pre-built Binaries

Installing a plugin takes three steps:

1. Download and install CyberEther from the [official website](https://cyberether.org).
2. Download the plugin bundles you want [here](https://stelline.space), such as Stelline or Blade.
3. Add the bundle through the CyberEther plugin manager, as described in [installing plugins](https://cyberether.org/docs/installing-plugins).

That's it. The bundle contains the plugin library, the manifest, and the example flowgraphs, so the blocks and examples appear in the flowgraph editor right away. Note that some blocks need extra host setup to actually run, as described in [host dependencies](/docs/host-dependencies).

## Container Images

The CyberEther CUDA images published to the GitHub Container Registry work as a complete runtime and development environment for Stelline:

```bash
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64-cuda
docker pull ghcr.io/luigifcruz/cyberether:ubuntu24-aarch64-cuda
```

Running the container needs more than plain GPU access, since the Stelline blocks reach into the host for hugepages, GPUDirect Storage, the display server, and the capture network:

```bash
$ nvidia_icd_json=$(find /usr/share /etc -path '*/vulkan/icd.d/nvidia_icd.json' -type f -print -quit 2>/dev/null | grep .) || (echo "nvidia_icd.json not found" >&2 && false)
$ sudo docker run -it --rm -u root \
    --net host \
    --privileged \
    --gpus=all \
    --cap-add CAP_SYS_PTRACE \
    --ipc=host \
    --volume /run/udev:/run/udev:ro \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --device=/dev/nvidia-fs* \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /mnt/huge:/mnt/huge \
    -v /mnt/raid:/mnt/raid \
    -v $nvidia_icd_json:$nvidia_icd_json:ro \
    -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
    -e DISPLAY=$DISPLAY \
    ghcr.io/luigifcruz/cyberether:ubuntu24-x86_64-cuda
```

The flags map to concrete block requirements. The `--gpus=all` and ICD mount give the container GPU compute and Vulkan graphics. The `--net host` flag exposes the capture interface for the [ATA Receiver](/docs/block-ata-receiver) multicast subscriptions, and `/mnt/huge` passes the hugepages mount it needs. The `/dev/nvidia-fs*` devices enable GPUDirect Storage for the writer blocks, with `/mnt/raid` standing in for your recording target.

## Build From Source

Every Stelline plugin is a standard Meson project. To build the central plugin, clone its repository and compile:

```bash
git clone https://github.com/luigifcruz/stelline
cd stelline
meson setup build
meson compile -C build
```

This produces `build/stelline.cep`, ready to be loaded by the CyberEther plugin manager. Userspace dependencies, including CyberEther itself, are resolved automatically as Meson subprojects, so only the host stack from [host dependencies](/docs/host-dependencies) has to be present on the machine.
