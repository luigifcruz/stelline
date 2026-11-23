#ifndef STELLINE_ATA_RECEIVER_BLOCK_HH
#define STELLINE_ATA_RECEIVER_BLOCK_HH

#include <string>
#include <vector>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct AtaReceiver : public Block::Config {
    std::string engine = "ibverbs";
    std::string interfaceAddress;
    U64 gpuDeviceId = 0;
    U64 masterCore = 0;
    std::vector<U64> workerCores = {};
    std::string subscriptions;
    std::vector<U64> totalBlock = {1, 1, 1024, 1};
    std::vector<U64> partialBlock = {1, 1, 1024, 1};
    std::vector<U64> offsetBlock = {0, 0, 0, 0};
    std::string dataType = "CF32";
    U64 packetsPerBurst = 0;
    U64 maxConcurrentBursts = 0;
    U64 maxConcurrentBlocks = 4;
    U64 outputPoolSize = 2;

    JST_BLOCK_TYPE(stelline_ata_receiver);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_NODE_SIZE(L);
    JST_BLOCK_PARAMS(engine, interfaceAddress, gpuDeviceId, masterCore, workerCores,
                     subscriptions, totalBlock, partialBlock, offsetBlock, dataType,
                     packetsPerBurst, maxConcurrentBursts, maxConcurrentBlocks,
                     outputPoolSize);
    JST_BLOCK_DESCRIPTION(
        "ATA Receiver",
        "Receives ATA voltage packets and assembles output tensors.",
        "# ATA Receiver\n"
        "\n"
        "Receives Allen Telescope Array voltage packets from a network card and "
        "assembles them into buffers of antennas, channels, samples, and polarizations."
        " It starts a Stelline chain, feeding a Correlator, a Beamformer, or a Slice "
        "ahead of a Channelizer.\n"
        "\n"
        "- **Only complete windows come out.** A buffer leaves once every packet of its"
        " time window has arrived.\n"
        "- **Late windows are dropped.** A window still missing packets is lost once "
        "newer data runs more than Max Concurrent Blocks windows ahead.\n"
        "- **Buffers start on the packet clock.** Each window begins at a timestamp "
        "that is a multiple of the Total Block samples.\n"
        "- **Samples keep their 8-bit scale.** Data Type CF32 converts the packet "
        "integers to floats without scaling them.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | Data Type, CF32 by default | Antennas x Channels x "
        "Samples x Polarizations from Total Block, with a `channelAxis` and a "
        "`sampleAxis`. Sets `timestamp` to the first sample of the window and "
        "`duration` to its samples, both in packet timestamp units. Neither "
        "`sampleRate` nor `frequency` is set. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Engine** | InfiniBandVerbs | InfiniBandVerbs | Network backend that "
        "receives the packets. |\n"
        "| **Interface Address** | Empty | PCI address of the network card | Card to "
        "receive on, such as 0000:01:00.0. Required. |\n"
        "| **GPU Device ID** | 0 | 0 to 65,535 | Index of the CUDA device that receives"
        " the payloads and assembles the buffers. |\n"
        "| **Master Core** | 0 | A CPU core index | Core of the receiver control "
        "thread. |\n"
        "| **Worker Cores** | Empty | One or more CPU core indices | Cores that poll "
        "the subscriptions, handed out in turn. Required. |\n"
        "| **Subscriptions** | Empty | One line per stream, as in Subscribe to Streams "
        "| Streams to receive. Required. |\n"
        "| **Total Block** | [1, 1, 1024, 1] | Four sizes, each a multiple of Partial "
        "Block | Output shape as antennas, channels, samples, and polarizations. "
        "Samples also sets the window length. |\n"
        "| **Partial Block** | [1, 1, 1024, 1] | Four sizes above 0 | Data in one "
        "packet, in the same order. It must match the sender, at 2 bytes per complex "
        "sample. |\n"
        "| **Offset Block** | [0, 0, 0, 0] | Antennas a multiple of Partial Block, the "
        "rest 0 | First antenna ID and channel number captured. Packets outside the "
        "window are dropped. |\n"
        "| **Data Type** | CF32 | CF32, CI8 | Output type. With CI8 the packet samples "
        "pass through as is. |\n"
        "| **Packets Per Burst** | 0 packets | Above 0 | Packets taken from the card at"
        " a time. Required. |\n"
        "| **Max Concurrent Bursts** | 0 bursts | Above 0 | Bursts held at once. "
        "Receive buffers total Packets Per Burst times this value, at least one per "
        "subscription. Required. |\n"
        "| **Max Concurrent Blocks** | 4 blocks | Above 0 | Time windows assembled at "
        "once. A window more than this many behind the newest is dropped. |\n"
        "| **Output Pool Size** | 2 buffers | Above 0 | Buffers shared by windows being"
        " assembled and finished ones waiting for the chain. When all are taken, new "
        "windows cannot start. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Bursts Usage** | Bursts in flight out of Max Concurrent Bursts. |\n"
        "| **Packets Received** | Packets placed into windows since the block started. "
        "|\n"
        "| **Tensors Usage** | Windows receiving (R) and assembling (C) out of Max "
        "Concurrent Blocks. |\n"
        "| **Ready Queue** | Finished buffers waiting for the chain, out of Output Pool"
        " Size. |\n"
        "| **Blocks Received** | Complete windows next to lost ones, as complete/lost. "
        "The bar shows the complete share. |\n"
        "| **Latest Timestamp** | Newest packet timestamp inside the antenna and "
        "channel window. |\n"
        "| **Throughput** | Incoming payload rate in Gbps, smoothed, counting dropped "
        "packets too. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume packets of Partial Block [1, 96, 16, 2] and the required "
        "network settings.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Full capture | Total Block [28, 192, 8192, 2], Offset Block [0, 352, 0, 0] |"
        " 28 antennas x 192 channels x 8,192 samples x 2 polarizations, complex, from "
        "28,672 packets, 672 MiB per buffer |\n"
        "| Smaller buffers | Same as Full capture, Data Type CI8 | The same shape as "
        "raw 8-bit samples, 168 MiB per buffer |\n"
        "| Lower latency | Total Block [28, 192, 1024, 2], Offset Block [0, 352, 0, 0] "
        "| 28 antennas x 192 channels x 1,024 samples x 2 polarizations, from 3,584 "
        "packets, 84 MiB per buffer |\n"
        "| One antenna | Total Block [1, 192, 8192, 2], Offset Block [5, 352, 0, 0] | "
        "Antenna ID 5 only, 192 channels x 8,192 samples x 2 polarizations, from 1,024 "
        "packets, 24 MiB per buffer |\n"
        "| Upper half of the band | Total Block [28, 96, 8192, 2], Offset Block [0, "
        "448, 0, 0] | Channels 448 to 543 of 28 antennas, 8,192 samples x 2 "
        "polarizations, from 14,336 packets, 336 MiB per buffer |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that a setting must be positive or not empty | Interface "
        "Address, Worker Cores, Subscriptions, Packets Per Burst, or Max Concurrent "
        "Bursts keeps its default | Set each of them. |\n"
        "| The block reports that it failed to configure the DAQIRI engine | The card, "
        "its driver, or the hugepages are missing, or Interface Address names another "
        "card | Check Setup and the PCI address of the card. |\n"
        "| Throughput rises but Packets Received stays at 0 | The packet antenna IDs or"
        " channel numbers fall outside Offset Block and Total Block | Set Offset Block "
        "to the first antenna and channel the senders use. |\n"
        "| Blocks Received counts only lost windows | Packets of each window never "
        "arrive, or Partial Block polarizations differ from Total Block | Subscribe to "
        "every sender in the window, and keep both polarization counts equal. |\n"
        "| Ready Queue stays full and lost windows rise | The chain reads buffers "
        "slower than they arrive | Lighten the chain, or raise Output Pool Size for "
        "short stalls. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| Linux | A CUDA 12 or newer toolkit and a CUDA build of CyberEther. |\n"
        "| Network card | A Mellanox ConnectX card, with libibverbs and libmlx5 from "
        "Mellanox OFED present when Stelline is built. |\n"
        "| Discrete GPU | The nvidia-peermem module, so the card writes payloads "
        "straight into GPU memory. Unified memory systems use pinned host memory "
        "instead. |\n"
        "| Hugepages | A hugepage pool for the packet headers. |\n"
        "\n"
        "## Reference\n"
        "\n"
        "### Subscribe to Streams\n"
        "\n"
        "Write one line per stream, from the sender to the destination, as below. Write"
        " `*` for any address or port, and an address without a port matches every "
        "port. Multicast destinations are joined on the card when the block starts.\n"
        "\n"
        "```\n"
        "- 192.0.2.10:10000 -> 239.0.0.1:50000\n"
        "- * -> 239.0.0.2:50000\n"
        "```\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Worker threads pull bursts of packets from the card, and each packet claims a "
        "slot in the time window named by its timestamp. A kernel copies every burst "
        "into place, and a window is handed on once all of its slots are filled. "
        "Finished buffers rotate through a fixed pool, so a slow chain loses windows "
        "instead of growing memory.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [ATA Receiver guide](https://stelline.space/docs/block-ata-receiver) "
        "covers the packet format and host setup.\n"
        "- **Correlator** forms visibilities from the voltages.\n"
        "- **Beamformer** combines antennas into beams.\n"
        "- **Nexus Bridge** supplies the metadata that recording writers need."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_ATA_RECEIVER_BLOCK_HH
