---
title: System Configuration
description: BIOS, kernel, and NIC settings required for high-throughput ingest.
order: 21
category: System Setup
---


It is important to validate that the server is configured correctly before running the Docker image. Below is a list of the configurations to check.

## BIOS Configuration

<!-- TODO: Write. -->

```
Bios> Advanced> PCIe/PCI/PnP Configuration> ASPM Support> Disabled
Bios> Advanced> Global C-State Control> Disabled
Bios> Advanced> ACPI Settings> ACPI SRAT L3 Cache As NUMA Domain > Disable
Bios> Advanced> North Bridge(NB) Configuration> ACS Enable> Disabled
Bios> Advanced> North Bridge(NB) Configuration> IOMMU> Disabled
```

### PCIe Resizable BAR

As previously discussed, this is not a hard requirement for Stelline but it's highly recommended for the best performance.

Most servers come with resizable BAR disabled by default. It's necessary to go to the BIOS to enable this feature. On Supermicro machines, navigate to Advanced > PCIe Configuration > \[Re-Size BAR Support > Enabled\] and \[Above 4G Decoding > Enabled\]. To validate if this worked, follow the verification instructions below.

Beware that servers with Broadcom PCIe switches do not support this feature. It's listed as supported, and even available in the server BIOS as an option, but it's actually not and will break the system. Check the [Validated Hardware](/docs/validated-hardware) page for more information about this issue.

To verify that the system and devices are properly configured, run the following command, replacing `41:00.0` with the correct device slot. Check the output for the "BAR 1" or "Region 1" size. If the reported size is greater than 256 MB, then Resizable BAR is enabled and functioning correctly.

```shell
$ sonata@dev-coyote1:~$ sudo lspci -s 41:00.0 -vvv | grep -E "BAR 1|Region 1"
Region 1: Memory at 2f000000000 (64-bit, prefetchable) [size=64G]
BAR 1: current size: 64GB, supported: 64MB 128MB 256MB 512MB 1GB 2GB 4GB 8GB 16GB 32GB 64GB
```

If there is any mismatch between these two values, it's an indication that something is wrong and further investigation is needed. We encountered such behavior while testing Supermicro servers with Broadcom PLX switches.

### ACS

BIOS >> Advanced >> NB Configuration >> ACS Enable >> Disabled

It's recommended to check the ACS state after booting up because some motherboards don't respect the BIOS settings. To automate this process on boot, create a file `/usr/local/sbin/acs-disable`:

```
#!/bin/bash
lspci -d "*:*:*" | awk '{print $1}' | while read -r BDF; do
  if sudo setpci -s "$BDF" ECAP_ACS+0x6.w > /dev/null 2>&1; then
    sudo setpci -s "$BDF" ECAP_ACS+0x6.w=0000
  fi
done
```

Now make the script executable:

```
$ sudo chmod +x /usr/local/sbin/acs-disable
```

Create a service calling the script automatically on boot in `/etc/systemd/system/acs-disable.service`.

```
[Unit]
Description=Disable PCIe ACS for GPU P2P
After=systemd-modules-load.service
Before=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/acs-disable
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

Enable the service.

```
$ sudo systemctl daemon-reload
$ sudo systemctl enable --now acs-disable
```

Use the command below to validate the ACS status. If all lines appears with `SrcValid-` instead of `SrcValid+` it means ACS is successfully disabled.

```
$ sudo lspci -vvv | grep ACSCtl
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
```

Now ACS should be disabled by default for every PCIe device on boot.

<!-- TODO: Write Multithreading section. -->

## ConnectX-7 PCIe Write Ordering

Required when a multi-port ConnectX-7 writes directly to separate GPUs.

The firmware on the validated ConnectX-7 cards reported `PCI_WR_ORDERING=per_mkey(0)`. On the validated AMD platform, this causes head-of-line blocking when the two functions of one physical card concurrently write to GPU memory registered under different MKeys. A single port can run at full rate while enabling the second port limits both receivers and causes physical ingress drops and pause frames.

Configure every ConnectX-7 PCI function to force PCIe relaxed ordering. The setting is stored per function, so configuring only the first function of a dual-port card is not sufficient.

First, start the Mellanox Software Tools service and list the device names:

```shell
$ sudo mst start
$ sudo mst status -v
```

Set `PCI_WR_ORDERING=1` on every ConnectX-7 function reported by `mst status`. For example, a system with two dual-port cards requires all four commands below:

```shell
$ sudo mlxconfig -y -d /dev/mst/mt4129_pciconf0 set PCI_WR_ORDERING=1
$ sudo mlxconfig -y -d /dev/mst/mt4129_pciconf0.1 set PCI_WR_ORDERING=1
$ sudo mlxconfig -y -d /dev/mst/mt4129_pciconf1 set PCI_WR_ORDERING=1
$ sudo mlxconfig -y -d /dev/mst/mt4129_pciconf1.1 set PCI_WR_ORDERING=1
```

The command should report `force_relax(1)` as the new value. This is a persistent firmware setting and does not become active until the host reboots. Stop active observations before rebooting.

```shell
$ sudo reboot
```

After the reboot, verify every function and unload the temporary MST drivers:

```shell
$ sudo mst start
$ for device in /dev/mst/mt4129_pciconf*; do
    sudo mlxconfig -d "$device" q PCI_WR_ORDERING
  done
$ sudo mst stop
```

Every query must report:

```text
PCI_WR_ORDERING                             force_relax(1)
```

On the validated four-port system, a minimal ATA Receiver and CUDA duplicate flowgraph delivered approximately 30 Gb/s per port with about 1.2 million physical drops per second under `per_mkey(0)`. With `force_relax(1)`, all four ports concurrently delivered 89.4 Gb/s each, 357.8 Gb/s aggregate, with zero physical drops, pause frames, or receive-buffer threshold events.

## Kernel Configuration

Some modules require kernel-level tuning to reach optimal performance, particularly in low-latency or high-throughput environments. This section outlines the relevant kernel arguments and explains when and why to use them.

Once you've identified which arguments apply to your system, append them in the `GRUB_CMDLINE_LINUX_DEFAULT` variable. After doing that, update the GRUB configuration and reboot the machine. After rebooting, use the instructions below to verify that the changes have been applied successfully.

```
$ sudo vim /etc/default/grub
$ sudo update-grub
$ sudo reboot
```

### Core Isolation

Required for `Transport` modules.

The Transport module in Stelline relies on strict latency guarantees to process NIC commands efficiently. To meet them, a subset of CPU cores must be isolated from general system activity. Isolating cores prevents the Linux scheduler from running background tasks, kernel threads, or interrupt handlers on those CPUs, ensuring they are fully dedicated to time-sensitive workloads.

First, it's important to carefully select which cores to isolate. This is straightforward in systems with a single socket, but it can get more complicated in dual-socket servers. If your system is the latter, just decide how many cores will be isolated and set the `isolcpus`, `nohz_full`, and `rcu_nocbs` arguments to match that range.

It's equally important to move all hardware interrupts off the isolated cores using the `irqaffinity` argument. This should cover the range of all remaining, non-isolated CPUs. For example, if the target machine has 8 cores and the first 3 are chosen for isolation, the kernel command line would look like this:

```
isolcpus=0-2 nohz_full=0-2 rcu_nocbs=0-2 irqaffinity=3-7 rcu_nocb_poll
```

If the system has multiple CPU sockets, it's important to isolate cores from **both sockets** to ensure balanced performance and consistent latency across NUMA domains. The command below shows how to identify the socket location of each logical core:

```
sonata@coyote2:~$ lscpu -e
CPU NODE SOCKET CORE
  0    0      0    0
  1    0      0    1
  2    0      0    2
  3    0      1    3
  4    0      1    4
  5    0      1    5
```

In this example, cores 0 to 2 belong to socket 0, and cores 3 to 5 belong to socket 1.

To isolate two cores from each socket (e.g., cores 0 and 1 from socket 0 and cores 3 and 4 from socket 1), the kernel arguments would be:

```
isolcpus=0-1,3-4 nohz_full=0-1,3-4 rcu_nocbs=0-1,3-4 irqaffinity=3,5 rcu_nocb_poll
```

This configuration ensures the selected cores are shielded from general scheduling and system noise, while remaining cores (in this case, 2 and 5) handle interrupts and background kernel activity.

### Disable IOMMU

Required for `Transport` and `I/O` modules.

<!-- TODO: Write why this is important. -->

For servers with an **AMD CPU**, use the following flags.

```
amd_iommu=off iommu=pt
```

If your server is equiped with an **Intel CPU**, use the following flags.

```
intel_iommu=off iommu=pt
```

After rebooting, verify if IOMMU is off by running the command below.

```
$ cat /proc/cmdline
... amd_iommu=off iommu=pt ...
```

### Disable PCIe Realloc

Required for `Transport` and `I/O` modules.

<!-- TODO: Write why this is important. -->

```
pci=realloc=off
```

After rebooting, verify if PCIe realloc is off by running the command below.

```
$ cat /proc/cmdline
... pci=realloc=off ...
```

### Huge Pages

Required for `Transport` modules.

The ATA Receiver uses hugepages for packet-header buffers registered with the ConnectX receive queues. These mappings reduce address translation overhead and provide predictable memory during high packet-rate ingest. An exhausted pool makes DAQIRI fall back to regular pages, which can reduce throughput and increase latency variation.

Use a pool of 256 pages with a 2 MiB page size. This reserves 512 MiB, which supports four simultaneous receivers with capacity left for additional mappings.

Add the following arguments to `GRUB_CMDLINE_LINUX_DEFAULT`:

```text
default_hugepagesz=2M hugepagesz=2M hugepages=256
```

Configure the hugetlbfs mount to use the same page size:

```shell
$ sudo mkdir -p /mnt/huge
$ sudo vim /etc/fstab
nodev /mnt/huge hugetlbfs pagesize=2M 0 0
$ sudo update-grub
$ sudo reboot
```

After rebooting, verify the total pool and page size:

```shell
$ grep -E 'HugePages_Total|HugePages_Free|HugePages_Rsvd|Hugepagesize' /proc/meminfo
HugePages_Total:     256
HugePages_Free:      256
HugePages_Rsvd:        0
Hugepagesize:       2048 kB
$ findmnt /mnt/huge -o TARGET,FSTYPE,OPTIONS
TARGET    FSTYPE    OPTIONS
/mnt/huge hugetlbfs rw,relatime,pagesize=2M
```

The free-page count can be lower if a receiver or another service has already started. On NUMA systems, also confirm that each node has enough pages for its local receivers:

```shell
$ for node in /sys/devices/system/node/node*; do
    printf '%s total=%s free=%s\n' \
      "$(basename "$node")" \
      "$(cat "$node/hugepages/hugepages-2048kB/nr_hugepages")" \
      "$(cat "$node/hugepages/hugepages-2048kB/free_hugepages")"
  done
```

Hugepages are selected when the receiver starts. If its log contains `MAP_HUGETLB allocation ... failed; falling back to regular pages`, correct the pool and restart the observation; a running receiver does not retry failed hugepage mappings.

### Miscellaneous Performance

Recommended for `Transport` modules.

These settings reduce background system noise and kernel-level overhead that can interfere with time-sensitive applications like the Transport and I/O modules. Disabling non-critical subsystems like auditing and soft lockup detection minimizes interruptions, while ensuring reliable timekeeping.

```
tsc=reliable audit=0 nosoftlockup
```

## Storage Configuration

<!-- TODO: Write. -->

### Software RAID

Create the software RAID.

```
sudo mdadm --create --verbose /dev/md0 --level=0 --raid-devices=2 /dev/nvme[5-8]n1
```

Sometimes drives from different RAID cards can be enumerated out of order. The command below can be used to verify which card each NVMe device is from.

```
$ udevadm info --query=all --name=/dev/nvme12n1 | grep DEVPATH
E: DEVPATH=/devices/pci0000:60/0000:60:01.1/0000:61:00.0/0000:62:04.0/0000:69:00.0/0000:6a:08.0/0000:6d:00.0/nvme/nvme12/nvme12n1
```

Format the partition as XFS filesystem.

```
sudo mkfs.xfs /dev/md0
```

Install XFS package.

```
sudo apt install xfsprogs
```

Mount the partition.

```
sudo mkdir -p /mnt/nvme-raid
sudo mount -t xfs /dev/md0 /mnt/nvme-raid
```
