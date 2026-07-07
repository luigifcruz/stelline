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

    JST_BLOCK_TYPE(ata_receiver);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(engine, interfaceAddress, gpuDeviceId, masterCore, workerCores,
                     subscriptions, totalBlock, partialBlock, offsetBlock, dataType,
                     packetsPerBurst, maxConcurrentBursts, maxConcurrentBlocks,
                     outputPoolSize);
    JST_BLOCK_DESCRIPTION(
        "ATA Receiver",
        "Receives ATA voltage packets and assembles output tensors.",
        "# ATA Receiver\n"
        "The ATA Receiver block ingests Allen Telescope Array voltage packets from a network "
        "interface and assembles them into dense output tensors. Dedicated worker threads "
        "receive packet bursts, a block map orders the fragments by timestamp, and a CUDA "
        "kernel gathers them into blocks shaped as [antennas, channels, samples, polarizations].\n\n"

        "## Arguments\n"
        "- **Engine**: Network backend used to receive packets (currently ibverbs).\n"
        "- **Interface Address**: Network interface address used to receive packets.\n"
        "- **GPU Device ID**: CUDA device used by the receiver backend.\n"
        "- **Master Core**: CPU core assigned to the receiver control thread.\n"
        "- **Worker Cores**: CPU cores assigned to networking workers.\n"
        "- **Subscriptions**: One 'source:port -> destination:port' subscription per line.\n"
        "- **Total Block**: Output block shape as [antennas, channels, samples, polarizations].\n"
        "- **Partial Block**: Per-packet fragment shape as [antennas, channels, samples, polarizations].\n"
        "- **Offset Block**: Input offset as [antennas, channels, samples, polarizations].\n"
        "- **Data Type**: Output tensor data type (CF32 or CI8).\n"
        "- **Packets Per Burst**: Maximum packets expected in each burst.\n"
        "- **Max Concurrent Bursts**: Maximum number of concurrent bursts in flight.\n"
        "- **Max Concurrent Blocks**: Maximum number of in-flight receive blocks.\n"
        "- **Output Pool Size**: Number of reusable output tensors for completed blocks.\n\n"

        "## Useful For\n"
        "- Ingesting high-rate voltage streams from the ATA network.\n"
        "- Feeding correlation or beamforming pipelines with assembled voltage blocks.\n"
        "- Monitoring packet loss, throughput, and block assembly health in real time.\n\n"

        "## Examples\n"
        "- Receive CF32 voltage blocks:\n"
        "  Config: Total Block=[28, 192, 8192, 2], Partial Block=[1, 96, 1024, 2]\n"
        "  Output: CF32[28, 192, 8192, 2] with a 'timestamp' attribute.\n\n"

        "## Implementation\n"
        "Network Bursts -> Block Map -> Gather Kernel -> Output Pool\n"
        "1. Worker threads receive packet bursts from the network interface.\n"
        "2. Each packet is filtered against the configured offset and mapped into a block by timestamp.\n"
        "3. A CUDA kernel gathers the packet payloads into the dense output tensor.\n"
        "4. Completed blocks are emitted from a reusable output pool with a timestamp attribute."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_ATA_RECEIVER_BLOCK_HH
