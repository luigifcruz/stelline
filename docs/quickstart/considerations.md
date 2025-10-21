# Considerations

Ingesting and processing hundreds of GB/s of data is a common problem that is usually best solved by utilizing hardware offloading techniques such as GPUDirect Storage and RDMA. These tools, for example, allow the program to offload the data transport from the CPU to specialized hardware. Although beneficial in increasing the performance, they can add complexity to the system. This project uses multiple of these leading edge technologies in some parts. Careful consideration is necessary to choose the right hardware configuration that takes complete advantage of such technologies. This section will cover the most common shortcomings.

## PCIe Resizable BAR

Resizable BAR (Base Address Register) is an important PCIe feature that allows the system to access more than 256 MB of a device’s memory in a single transaction. Normally, GPUs and other PCIe devices are limited to exposing only 256 MB of memory at a time, which can create bottlenecks. Enabling Resizable BAR lifts this restriction, allowing for more efficient use of the full memory space.

// TODO: Motherboard support.

## PCIe 5.0

Although this is not a hard requirement, it’s recommended to use a server that has PCIe 5.0 support. Bottlenecks are most often caused by interconnect interfaces in the majority of the data patterns supported by this project.

## Storage Configuration

The third and final piece of real-time scientific data processing is storing the output data from the computation step. Depending on the application, this output can range from megabytes to gigabytes of data per second. On the lower end of bandwidth, simple solutions like a single NVMe device or even hard-drivers should be more than enough to meet requirements. On another hand, storing multiple gigabytes of data per second can be hard necessitating specialized hardware such as multiple NVMe in RAID0 or RAMFS. Let's talk about the pros and cons of the main options.

### NVMe

// TODO: Write.

### RAM Disk

// TODO: Write.

## PCIe Topology

// TODO: Write.
