#include "daqiri_config.hh"

#include <string>

#include <jetstream/logger.hh>
#include <jetstream/macros.hh>

#include "packet.hh"

namespace Jetstream::Modules {

using namespace stelline::domains::stelline::utils;

Result BuildDaqiriRxConfig(const DaqiriRxConfigParams& params,
                           const std::vector<SubscriptionEndpoint>& subscriptions,
                           daqiri::NetworkConfig& cfg) {
    const auto subscriptionCount = subscriptions.size();
    const auto totalNumBufs = static_cast<size_t>(params.packetsPerBurst * params.maxConcurrentBursts);

    cfg.log_level_ = daqiri::LogLevel::TRACE;
    cfg.tx_meta_buffers_ = daqiri::DEFAULT_TX_META_BUFFERS * 8;
    cfg.rx_meta_buffers_ = daqiri::DEFAULT_RX_META_BUFFERS * 8;

    cfg.common_.version = 1;
    cfg.common_.master_core_ = params.masterCore;
    cfg.common_.dir = daqiri::Direction::RX;
    cfg.common_.loopback_ = daqiri::LoopbackType::DISABLED;
    cfg.common_.stream_type = daqiri::StreamType::RAW;
    cfg.common_.engine = daqiri::EngineType::IBVERBS;

    daqiri::InterfaceConfig interfaceCfg = {};
    interfaceCfg.address_ = params.interfaceAddress;
    interfaceCfg.rx_.flow_isolation_ = true;

    U16 nextId = 0;
    for (const auto& subscription : subscriptions) {
        const auto queueNumBufs = totalNumBufs / subscriptionCount +
                                  (static_cast<size_t>(nextId) < (totalNumBufs % subscriptionCount) ? 1 : 0);
        if (queueNumBufs == 0) {
            JST_ERROR("[MODULE_ATA_RECEIVER] Total buffer count is too small for the number of subscriptions.");
            return Result::ERROR;
        }

        const auto headerMrName = "RX_HEADER_" + std::to_string(nextId);
        const auto dataMrName = "RX_DATA_" + std::to_string(nextId);

        daqiri::MemoryRegionConfig headerMemoryCfg = {};
        headerMemoryCfg.name_ = headerMrName;
        headerMemoryCfg.kind_ = daqiri::MemoryKind::HUGE;
        headerMemoryCfg.affinity_ = 0;
        headerMemoryCfg.buf_size_ = kPacketHeaderOffset + kPacketHeaderSize;
        headerMemoryCfg.num_bufs_ = queueNumBufs;
        headerMemoryCfg.access_ = daqiri::MEM_ACCESS_LOCAL;
        headerMemoryCfg.owned_ = true;
        cfg.mrs_.emplace(headerMemoryCfg.name_, headerMemoryCfg);

        daqiri::MemoryRegionConfig dataMemoryCfg = {};
        dataMemoryCfg.name_ = dataMrName;
        dataMemoryCfg.kind_ = params.dataMemoryKind;
        dataMemoryCfg.affinity_ = params.gpuDeviceId;
        dataMemoryCfg.buf_size_ = kPacketDataSize;
        dataMemoryCfg.num_bufs_ = queueNumBufs;
        dataMemoryCfg.access_ = daqiri::MEM_ACCESS_LOCAL;
        dataMemoryCfg.owned_ = true;
        cfg.mrs_.emplace(dataMemoryCfg.name_, dataMemoryCfg);

        daqiri::RxQueueConfig queueCfg = {};
        queueCfg.common_.name_ = "subscription-" + std::to_string(nextId);
        queueCfg.common_.id_ = nextId;
        queueCfg.common_.batch_size_ = params.packetsPerBurst;
        queueCfg.common_.cpu_core_ = jst::fmt::format("{}", params.workerCores[nextId % params.workerCores.size()]);
        queueCfg.common_.mrs_ = {headerMrName, dataMrName};
        queueCfg.timeout_us_ = 0;
        interfaceCfg.rx_.queues_.push_back(queueCfg);

        EndpointMatch source;
        JST_CHECK(ParseEndpoint(subscription.source, source));

        EndpointMatch destination;
        JST_CHECK(ParseEndpoint(subscription.destination, destination));

        daqiri::FlowConfig flowCfg = {};
        flowCfg.name_ = "subscription-" + std::to_string(nextId);
        flowCfg.id_ = nextId;
        flowCfg.action_.type_ = daqiri::FlowType::QUEUE;
        flowCfg.action_.id_ = nextId;
        flowCfg.match_.type_ = daqiri::FlowMatchType::IPV4_UDP;
        flowCfg.match_.udp_src_ = source.hasPort ? source.port : 0;
        flowCfg.match_.udp_dst_ = destination.hasPort ? destination.port : 0;
        flowCfg.match_.ipv4_len_ = 0;
        flowCfg.match_.ipv4_src_ = source.hasIp ? source.ip : INADDR_ANY;
        flowCfg.match_.ipv4_dst_ = destination.hasIp ? destination.ip : INADDR_ANY;
        interfaceCfg.rx_.flows_.push_back(flowCfg);

        nextId += 1;
    }

    cfg.ifs_.push_back(interfaceCfg);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
