#ifndef STELLINE_BITS_TRANSPORT_BASE_HH
#define STELLINE_BITS_TRANSPORT_BASE_HH

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/advanced_network/adv_network_rx.h>

#include <stelline/helpers.hh>
#include <stelline/yaml/types/block_shape.hh>
#include <stelline/operators/transport/base.hh>

namespace stelline::bits::transport {

inline BitInterface TransportBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::transport;

    // Fetch configuration YAML.

    auto total_block = FetchNodeArg<BlockShape>(app, config, "total_block");
    auto partial_block = FetchNodeArg<BlockShape>(app, config, "partial_block");
    auto offset_block = FetchNodeArg<BlockShape>(app, config, "offset_block");
    auto concurrent_blocks = FetchNodeArg<uint64_t>(app, config, "concurrent_blocks");
    auto output_pool_size = FetchNodeArg<uint64_t>(app, config, "output_pool_size");
    auto enable_csv_logging = FetchNodeArg<bool>(app, config, "enable_csv_logging", false);
    auto sorter_depth = FetchNodeArg<uint64_t>(app, config, "sorter_depth", 16);

    auto rdma_gpu = FetchNodeArg<uint64_t>(app, config, "rdma_gpu");
    auto rdma_nic = FetchNodeArg<std::string>(app, config, "rdma_nic");
    auto rdma_master_core = FetchNodeArg<uint64_t>(app, config, "rdma_master_core");
    auto rdma_worker_cores = FetchNodeArg<std::vector<uint64_t>>(app, config, "rdma_worker_cores");
    auto rdma_max_bursts = FetchNodeArg<uint64_t>(app, config, "rdma_max_bursts");
    auto rdma_burst_size = FetchNodeArg<uint64_t>(app, config, "rdma_burst_size");

    HOLOSCAN_LOG_INFO("Transport Configuration:");
    HOLOSCAN_LOG_INFO("  RDMA:");
    HOLOSCAN_LOG_INFO("    GPU: {}", rdma_gpu);
    HOLOSCAN_LOG_INFO("    NIC: {}", rdma_nic);
    HOLOSCAN_LOG_INFO("    Master Core: {}", rdma_master_core);
    HOLOSCAN_LOG_INFO("    Worker Cores: {}", rdma_worker_cores);
    HOLOSCAN_LOG_INFO("    Max Bursts: {}", rdma_max_bursts);
    HOLOSCAN_LOG_INFO("    Burst Size: {}", rdma_burst_size);
    HOLOSCAN_LOG_INFO("  Concurrent Blocks: {}", concurrent_blocks);
    HOLOSCAN_LOG_INFO("  Output Pool Size: {}", output_pool_size);
    HOLOSCAN_LOG_INFO("  Enable CSV Logging: {}", enable_csv_logging);
    HOLOSCAN_LOG_INFO("  Total Block: {}", total_block);
    HOLOSCAN_LOG_INFO("  Partial Block: {}", partial_block);
    HOLOSCAN_LOG_INFO("  Offset Block: {}", offset_block);
    HOLOSCAN_LOG_INFO("  Sorter Depth: {}", sorter_depth);

    // Build RDMA configuration.

    ops::AdvNetConfigYaml ano_cfg = {};
    ano_cfg.common_.version = 1;
    ano_cfg.common_.master_core_ = rdma_master_core;
    ano_cfg.common_.dir = ops::AdvNetDirection::RX;

    ops::AdvNetRxConfig ano_cfg_interface = {};
    ano_cfg_interface.if_name_ = rdma_nic;
    ano_cfg_interface.port_id_ = 0;
    ano_cfg_interface.flow_isolation_ = true;

    ops::RxQueueConfig ano_cfg_rx_queue = {};
    ano_cfg_rx_queue.common_.name_ = "adc_rx";
    ano_cfg_rx_queue.common_.id_ = 0;
    ano_cfg_rx_queue.common_.gpu_dev_ = rdma_gpu;
    ano_cfg_rx_queue.common_.gpu_direct_ = true;
    ano_cfg_rx_queue.common_.hds_ = ReceiverOp::TransportHeaderSize +
                                    ReceiverOp::VoltageHeaderSize;
    ano_cfg_rx_queue.common_.cpu_cores_ = fmt::format("{}", fmt::join(rdma_worker_cores, ","));
    ano_cfg_rx_queue.common_.max_packet_size_ = ReceiverOp::TransportHeaderSize +
                                                ReceiverOp::VoltageHeaderSize +
                                                ReceiverOp::VoltageDataSize;
    ano_cfg_rx_queue.common_.num_concurrent_batches_ = rdma_max_bursts;
    ano_cfg_rx_queue.common_.batch_size_ = rdma_burst_size;
    ano_cfg_rx_queue.output_port_ = "bench_rx_out";

    ano_cfg_interface.queues_.push_back(ano_cfg_rx_queue);

    ops::FlowConfig ano_cfg_flow = {};
    ano_cfg_flow.name_ = "adc_rx";
    ano_cfg_flow.action_.type_ = ops::FlowType::QUEUE;
    ano_cfg_flow.action_.id_ = 0;
    ano_cfg_flow.match_.udp_src_ = 10000;
    ano_cfg_flow.match_.udp_dst_ = 50000;

    ano_cfg_interface.flows_.push_back(ano_cfg_flow);

    ano_cfg.rx_.push_back(ano_cfg_interface);

    // Instantiate operators.

    auto ano_rx = app->template make_operator<ops::AdvNetworkOpRx>(
        fmt::format("ano_rx_{}", id),
        Arg("cfg", ano_cfg),
        app->template make_condition<BooleanCondition>("is_alive", true)
    );

    auto receiver = app->template make_operator<ReceiverOp>(
        fmt::format("receiver_{}", id),
        Arg("concurrent_blocks", concurrent_blocks),
        Arg("total_block", total_block),
        Arg("partial_block", partial_block),
        Arg("offset_block", offset_block),
        Arg("output_pool_size", output_pool_size),
        Arg("enable_csv_logging", enable_csv_logging)
    );

    auto sorter = app->template make_operator<SorterOp>(
        fmt::format("sorter_{}", id),
        Arg("depth", sorter_depth)
    );

    // Connect operators.

    app->add_flow(ano_rx, receiver, {{"bench_rx_out", "burst_in"}});
    app->add_flow(receiver, sorter, {{"dsp_block_out", "dsp_block_in"}});

    return {sorter, sorter};
}

}  // namespace stelline::bits::transport

#endif  // STELLINE_BITS_TRANSPORT_BASE_HH
