# Dependencies

After finishing the machine configuration, some host dependencies need to be installed before running the Docker container. Some of these are not required. Refer to the Stelline module documentation to check whether any of these components are necessary for your specific requirements.

## NVIDIA Driver & CUDA

Install the latest NVIDIA Driver and the latest CUDA release following the official instructions available [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_network). We recommend using the network installation method and the open-source kernel module because the latest versions of GPUDirect doesn't support the proprietary kernel module. Reboot the machine and run `nvidia-smi` to verify if all the GPUs installed are enumerated.

```
$ sonata@dev-coyote1:~$ nvidia-smi
Tue May  6 20:45:58 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:41:00.0 Off |                  Off |
| 30%   25C    P8              7W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:61:00.0 Off |                  Off |
| 30%   24C    P8             12W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## NVIDIA Docker Container Toolkit

This is required to use the GPU and other driver utilities from inside the Docker container.

First, install the the Docker package following the official Docker instructions available [here](https://docs.docker.com/engine/install/ubuntu/). Don't forget to add your user to the Docker permission group.

```
$ sudo groupadd docker
$ sudo usermod -aG docker $USER
```

Now that Docker is installed, follow the instructions in the NVIDIA official documentation available [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Make sure to set the runtime and reload the Docker daemon. Run the command below to validate if the package was installed correctly. The available GPUs should be listed by the `nvidia-smi` command from inside the container.

```
$ docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu20.04 nvidia-smi
Unable to find image 'nvidia/cuda:12.3.0-base-ubuntu20.04' locally
12.3.0-base-ubuntu20.04: Pulling from nvidia/cuda
96d54c3075c9: Pull complete
f35fac6f9729: Pull complete
c16243f33326: Pull complete
752b1f8b6764: Pull complete
7d4f0f8effa7: Pull complete
Digest: sha256:eaa6ecb974689231ed32dc842281ff0a86b69bcfb4a7568ea7093372a00cdbf2
Status: Downloaded newer image for nvidia/cuda:12.3.0-base-ubuntu20.04
Wed May  7 01:18:31 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.51.03              Driver Version: 575.51.03      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:41:00.0 Off |                  Off |
| 30%   26C    P8              6W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX 6000 Ada Gene...    Off |   00000000:61:00.0 Off |                  Off |
| 30%   28C    P8             25W /  300W |       2MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## Mellanox OFED

This package is required by some of the Stelline networking modules.

We recommend using a ConnectX-7 or newer for best compatibility and feature set. It's theoretically possible to use cards from other manufacturers but you might run into incompatibilities or missing features.

The first step is go to the [official NVIDIA website](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/) and download the latest MLNX_OFED tarball file available. We're planning to move towards the NVIDIA DOCA in future releases.

```
$ sudo apt install wget
$ wget [REPLACE_WITH_DOWNLOAD_URL]
```

After downloading the installed, uncompress it in the current directory.

```
$ tar -zxvf MLNX_OFED_LINUX*.tgz
$ cd MLNX*
```

Navigate to the uncompressed OFED package and run the following command. The arguments enable extra features such as GPUDirect Storage to the NVMe. Note that this will automatically upgrade the firmware of all ConnectX cards installed to the latest version.

```
$ sudo ./mlnxofedinstall --all --with-nvmf --with-nfsrdma --force-dkms
```

Now reconfigure the GPU driver kernel modules to ensure it's aware of the new OFED installation.

```
$ nv_driver_version=$(modinfo nvidia | awk '/^version:/ {print $2}' | cut -d. -f1)
$ sudo dpkg-reconfigure nvidia-dkms-$nv_driver_version-open
```

Enable `nvidia-peermem` on boot.

```
$ echo "nvidia-peermem" | sudo tee /etc/modules-load.d/nvidia-peermem.conf
```

Since we just installed some kernel modules, we should update RAMFS and reboot the machine is necessary before proceeding.

```
$ sudo /etc/init.d/openibd restart
$ sudo update-initramfs -u -k `uname -r`
$ sudo reboot
```

After rebooting, the installation can be verified by running `ibv_devinfo`. Information about the install network interfaces should be printed. This only verifies the network drivers installation. To validate the NVMe RDMA modules go to the "GPUDirect Storage" section.

```
$ ibv_devinfo
hca_id: mlx5_1
        transport:                      InfiniBand (0)
        fw_ver:                         28.43.2566
        node_guid:                      b8e9:2403:004e:de81
        sys_image_guid:                 b8e9:2403:004e:de80
        vendor_id:                      0x02c9
        vendor_part_id:                 4129
        hw_ver:                         0x0
        board_id:                       MT_0000000834
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
```

Another way to verify that the drivers were installed correctly is to check if they were loaded by the kernel using the `lsmod` command.

```
$ sudo lsmod | grep ib_core
ib_core               524288  8 rdma_cm,ib_ipoib,iw_cm,ib_umad,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm
mlx_compat             20480  14 rdma_cm,ib_ipoib,mlxdevm,nvme,mlxfw,iw_cm,nvme_core,ib_umad,ib_core,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm,mlx5_core
```

## GPUDirect Storage

This installation is required by some of the Stelline I/O modules.

Make sure "Mellanox OFED" is installed and IOMMU is disabled before proceeding with the GPUDirect Storage installation.

First, install the kernel drivers of GPUDirect Storage. A reboot is necessary after this step.

IMPORTANT: Do not install version `2.25.6` of `nvidia-fs`. This is known to be buggy causing memory mapping issues as described [here](https://github.com/NVIDIA/gds-nvidia-fs/issues/60).

```
$ sudo apt-get install nvidia-fs nvidia-gds
$ sudo reboot
```

After rebooting, it’s advisable to disable GDS compatibility mode. This feature, which is enabled by default, causes I/O operations to be routed through the host memory if DMA is not supported by the current installation. While this is usually desirable, it could allow a misconfiguration to go unnoticed.

```
$ sudo vim /etc/cufile.json  # Edit `properties.allow_compat_mode: false`.
```

To validate the GPUDirect Storage installation, check if the kernel modules are running by running the command below. If everything was installed correctly, expect to see `nvme_core` listed both under `nvme_auth` and `mlx_compat`. The `nvidia_fs` kernel module should also be loaded.

```
$ lsmod | grep nvme_core
nvme_core             200704  1 nvme
nvme_auth              28672  1 nvme_core
mlx_compat             20480  14 rdma_cm,ib_ipoib,mlxdevm,nvme,mlxfw,iw_cm,nvme_core,ib_umad,ib_core,rdma_ucm,ib_uverbs,mlx5_ib,ib_cm,mlx5_core
$ lsmod | grep nvidia_fs
nvidia_fs             258048  0
nvidia              11427840  4 nvidia_uvm,nvidia_peermem,nvidia_fs,nvidia_modeset
```

Now run the `gdscheck` script and check if the NVMe driver is marked as `Supported`.

```
$ /usr/local/cuda/gds/tools/gdscheck -p
GDS release version: 1.14.0.30
nvidia_fs version:  2.25 libcufile version: 2.12
Platform: x86_64
============
ENVIRONMENT:
============
=====================
DRIVER CONFIGURATION:
=====================
NVMe P2PDMA        : Unsupported
NVMe               : Supported
NVMeOF             : Unsupported
SCSI               : Unsupported
ScaleFlux CSD      : Unsupported
NVMesh             : Unsupported
DDN EXAScaler      : Unsupported
IBM Spectrum Scale : Unsupported
NFS                : Unsupported
BeeGFS             : Unsupported
ScaTeFS            : Unsupported
WekaFS             : Unsupported
Userspace RDMA     : Unsupported
--Mellanox PeerDirect : Disabled
--rdma library        : Not Loaded (libcufile_rdma.so)
--rdma devices        : Not configured
--rdma_device_status  : Up: 0 Down: 0
```

Finally, to make sure everything is working and configured correctly, use `gdsio` to write a file to the XFS mount using GPUDirect Storage.

```
$ sudo /usr/local/cuda/gds/tools/gdsio -f /mnt/nvme-raid/test.bin -x 0 -d 0 -s 5G -I 1
IoType: WRITE XferType: GPUD Threads: 1 DataSetSize: 5242880/5242880(KiB) IOSize: 1024(KiB) Throughput: 7.473484 GiB/sec, Avg_Latency: 130.389453 usecs ops: 5120 total_time 0.669032 secs
```

Additionally, the `nvidia-fs` can be checked to validate if the last command ran correctly. If so, the `Mmap` and `Bar1-map` should both list `n=1 ok=1` without any errors `err=0`.

```
$ cat /proc/driver/nvidia-fs/stats
GDS Version: 1.14.0.31
NVFS statistics(ver: 4.0)
NVFS Driver(version: 2.25.7)
Mellanox PeerDirect Supported: True
IO stats: Disabled, peer IO stats: Disabled
Logging level: info

Active Shadow-Buffer (MiB): 0
Active Process: 0
Reads                : err=0 io_state_err=0
Sparse Reads         : n=0 io=0 holes=0 pages=0
Writes               : err=0 io_state_err=0 pg-cache=0 pg-cache-fail=0 pg-cache-eio=0
Mmap                 : n=1 ok=1 err=0 munmap=2
Bar1-map             : n=1 ok=1 err=0 free=1 callbacks=0 active=0 delay-frees=0
Error                : cpu-gpu-pages=0 sg-ext=0 dma-map=0 dma-ref=0
Ops                  : Read=0 Write=0 BatchIO=0
GPU 0000:01:00.0  uuid:fb1e6550-2a78-d767-1f0d-7512eb8c552d : Registered_MiB=0 Cache_MiB=0 max_pinned_MiB=1
```
