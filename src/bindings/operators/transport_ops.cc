#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <arpa/inet.h>

#include <stdexcept>
#include <string>
#include <vector>

#include <holoscan/core/application.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/transport/base.hh>
#include <stelline/system_info.hh>
#include <stelline/yaml/types/block_shape.hh>

#include <advanced_network/types.h>
#include <advanced_network/common.h>

namespace py = pybind11;
using namespace stelline::operators::transport;
using namespace holoscan;

namespace {

struct EndpointMatch {
    bool has_ip = false;
    bool has_port = false;
    in_addr_t ip = INADDR_ANY;
    uint16_t port = 0;
};

EndpointMatch parse_endpoint(const std::string& endpoint) {
    EndpointMatch match;

    const auto separator = endpoint.rfind(':');
    const bool has_port = separator != std::string::npos;

    const std::string ip_text = has_port ? endpoint.substr(0, separator) : endpoint;
    const std::string port_text = has_port ? endpoint.substr(separator + 1) : "*";

    if (!ip_text.empty() && ip_text != "*") {
        in_addr addr = {};
        if (inet_pton(AF_INET, ip_text.c_str(), &addr) != 1) {
            throw std::runtime_error("Invalid IPv4 endpoint: " + endpoint);
        }
        match.has_ip = true;
        match.ip = addr.s_addr;
    }

    if (!port_text.empty() && port_text != "*") {
        int port = 0;
        try {
            port = std::stoi(port_text);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid UDP port in endpoint: " + endpoint);
        }

        if (port < 0 || port > 65535) {
            throw std::runtime_error("UDP port out of range in endpoint: " + endpoint);
        }

        match.has_port = true;
        match.port = static_cast<uint16_t>(port);
    }

    return match;
}

EndpointMatch parse_subscription_endpoint(const py::handle& item, const char* key) {
    const auto subscription = py::reinterpret_borrow<py::dict>(item);
    const auto key_obj = py::str(key);

    if (!subscription.contains(key_obj)) {
        throw std::runtime_error(std::string("Each subscription must define '") + key + "'.");
    }

    return parse_endpoint(subscription[key_obj].cast<std::string>());
}

}  // namespace

class PyAtaReceiverOp : public AtaReceiverOp {
public:
    using AtaReceiverOp::AtaReceiverOp;

    PyAtaReceiverOp(Fragment* fragment,
                    const py::args& args,
                    int gpu_device_id,
                    const std::string& interface_address,
                    int master_core,
                    const std::vector<int>& worker_cores,
                    const std::string& packet_parser_type,
                    uint64_t packet_header_offset,
                    uint64_t packet_header_size,
                    int packet_data_size,
                    uint64_t packets_per_burst,
                    uint64_t max_concurrent_bursts,
                    uint64_t max_concurrent_blocks,
                    py::object subscriptions,
                    const stelline::BlockShape& total_block,
                    const stelline::BlockShape& partial_block,
                    const stelline::BlockShape& offset_block,
                    uint64_t output_pool_size,
                    bool enable_csv_logging,
                    const std::string& dtype,
                    const std::string& name = "ata_receiver")
        : AtaReceiverOp(ArgList{Arg("max_concurrent_blocks", max_concurrent_blocks),
                                Arg("total_block", total_block),
                                Arg("packet_header_size", packet_header_size),
                                Arg("packet_header_offset", packet_header_offset),
                                Arg("partial_block", partial_block),
                                Arg("offset_block", offset_block),
                                Arg("output_pool_size", output_pool_size),
                                Arg("enable_csv_logging", enable_csv_logging),
                                Arg("dtype", dtype)}) {
        // Configure ANO.

        {
            using namespace holoscan::advanced_network;

            NetworkConfig cfg = {};
            const auto subscription_count = static_cast<size_t>(py::len(subscriptions));
            const auto total_num_bufs = static_cast<size_t>(packets_per_burst * max_concurrent_bursts);

            cfg.log_level_ =  holoscan::advanced_network::LogLevel::TRACE;
            cfg.tx_meta_buffers_ = DEFAULT_TX_META_BUFFERS;
            cfg.rx_meta_buffers_ = DEFAULT_RX_META_BUFFERS;

            cfg.common_.version = 1;
            cfg.common_.master_core_ = master_core;
            cfg.common_.dir = Direction::RX;
            cfg.common_.manager_type = ManagerType::DPDK;
            cfg.common_.loopback_ = LoopbackType::DISABLED;

            const auto& sys = stelline::SystemInfo::instance();

            {
                InterfaceConfig interface_cfg = {};

                interface_cfg.address_ = interface_address;
                interface_cfg.rx_.flow_isolation_ = true;

                {
                    uint16_t next_id = 0;

                    for (const auto& item : subscriptions.cast<py::iterable>()) {
                        const auto queue_num_bufs = total_num_bufs / subscription_count +
                                                    (static_cast<size_t>(next_id) < (total_num_bufs % subscription_count) ? 1 : 0);
                        const auto header_mr_name = "RX_HEADER_" + std::to_string(next_id);
                        const auto data_mr_name = "RX_DATA_" + std::to_string(next_id);
                        const auto source = parse_subscription_endpoint(item, "source");
                        const auto destination = parse_subscription_endpoint(item, "destination");

                        if (queue_num_bufs == 0) {
                            throw std::runtime_error("Total buffer count is too small for the number of subscriptions.");
                        }

                        {
                            MemoryRegionConfig memory_cfg = {};

                            memory_cfg.name_ = header_mr_name;
                            memory_cfg.kind_ = MemoryKind::HUGE;
                            memory_cfg.affinity_ = 0;
                            memory_cfg.buf_size_ = packet_header_offset + packet_header_size;
                            memory_cfg.num_bufs_ = queue_num_bufs;
                            memory_cfg.access_ = MEM_ACCESS_LOCAL;
                            memory_cfg.owned_ = true;

                            cfg.mrs_.emplace(memory_cfg.name_, memory_cfg);
                        }

                        {
                            MemoryRegionConfig memory_cfg = {};

                            memory_cfg.name_ = data_mr_name;
                            memory_cfg.kind_ = sys.unifiedMemory() ? MemoryKind::HOST_PINNED :
                                                                     MemoryKind::DEVICE;
                            memory_cfg.affinity_ = gpu_device_id;
                            memory_cfg.buf_size_ = packet_data_size;
                            memory_cfg.num_bufs_ = queue_num_bufs;
                            memory_cfg.access_ = MEM_ACCESS_LOCAL;
                            memory_cfg.owned_ = true;

                            cfg.mrs_.emplace(memory_cfg.name_, memory_cfg);
                        }

                        RxQueueConfig queue_cfg = {};
                        queue_cfg.common_.name_ = "subscription-" + std::to_string(next_id);
                        queue_cfg.common_.id_ = next_id;
                        queue_cfg.common_.batch_size_ = packets_per_burst;
                        queue_cfg.common_.cpu_core_ = fmt::format("{}", worker_cores[next_id % worker_cores.size()]);
                        queue_cfg.common_.mrs_ = {header_mr_name, data_mr_name};
                        queue_cfg.timeout_us_ = 0;
                        interface_cfg.rx_.queues_.push_back(queue_cfg);

                        FlowConfig flow_cfg = {};
                        flow_cfg.name_ = "subscription-" + std::to_string(next_id);
                        flow_cfg.id_ = next_id;
                        flow_cfg.action_.type_ = FlowType::QUEUE;
                        flow_cfg.action_.id_ = next_id;
                        flow_cfg.match_.type_ = FlowMatchType::NORMAL;
                        flow_cfg.match_.udp_src_ = source.has_port ? source.port : 0;
                        flow_cfg.match_.udp_dst_ = destination.has_port ? destination.port : 0;
                        flow_cfg.match_.ipv4_len_ = 0;
                        flow_cfg.match_.ipv4_src_ = source.has_ip ? source.ip : INADDR_ANY;
                        flow_cfg.match_.ipv4_dst_ = destination.has_ip ? destination.ip : INADDR_ANY;
                        interface_cfg.rx_.flows_.push_back(flow_cfg);

                        next_id += 1;
                    }
                }

                cfg.ifs_.push_back(interface_cfg);
            }

            if (adv_net_init(cfg) != Status::SUCCESS) {
                HOLOSCAN_LOG_ERROR("Failed to configure the Advanced Network manager.");
                throw std::runtime_error("Module initialization failed.");
            }
            HOLOSCAN_LOG_INFO("Configured the Advanced Network manager");
        }

        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PyDummyReceiverOp : public DummyReceiverOp {
public:
    using DummyReceiverOp::DummyReceiverOp;

    PyDummyReceiverOp(Fragment* fragment,
                      const py::args& args,
                      const stelline::BlockShape& total_block,
                      const std::string& name = "dummy_receiver")
        : DummyReceiverOp(ArgList{Arg("total_block", total_block)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

class PySorterOp : public SorterOp {
public:
    using SorterOp::SorterOp;

    PySorterOp(Fragment* fragment,
               const py::args& args,
               uint64_t depth,
               const std::string& name = "sorter")
        : SorterOp(ArgList{Arg("depth", depth)}) {
        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_transport_ops, m) {
    m.doc() = "Stelline transport operators module";

    py::class_<AtaReceiverOp, PyAtaReceiverOp, Operator, std::shared_ptr<AtaReceiverOp>>(m, "AtaReceiverOp")
        .def(py::init<Fragment*,
                      const py::args&,
                      int,
                      const std::string&,
                      int,
                      const std::vector<int>&,
                      const std::string&,
                      int,
                      int,
                      int,
                      uint64_t,
                      uint64_t,
                      uint64_t,
                      py::object,
                      const stelline::BlockShape&,
                      const stelline::BlockShape&,
                      const stelline::BlockShape&,
                      uint64_t,
                      bool,
                      const std::string&,
                      const std::string&>(),
              py::arg("fragment"),
              py::arg("gpu_device_id"),
              py::arg("interface_address"),
              py::arg("master_core"),
              py::arg("worker_cores"),
              py::arg("packet_parser_type"),
              py::arg("packet_header_offset"),
              py::arg("packet_header_size"),
              py::arg("packet_data_size"),
              py::arg("packets_per_burst"),
              py::arg("max_concurrent_bursts"),
              py::arg("max_concurrent_blocks"),
              py::arg("subscriptions"),
              py::arg("total_block"),
              py::arg("partial_block"),
              py::arg("offset_block"),
              py::arg("output_pool_size"),
              py::arg("enable_csv_logging"),
              py::arg("dtype"),
              py::arg("name") = "ata_receiver")
        .def("tick", &AtaReceiverOp::tick)
        .def("format_metrics", &AtaReceiverOp::formatMetrics)
        .def("set_manifest_provider", &AtaReceiverOp::setManifestProvider)
        .def("set_metrics_provider", &AtaReceiverOp::setMetricsProvider)
        .def_property_readonly("manifest", &AtaReceiverOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &AtaReceiverOp::metrics, py::return_value_policy::reference);

    py::class_<DummyReceiverOp, PyDummyReceiverOp, Operator, std::shared_ptr<DummyReceiverOp>>(m, "DummyReceiverOp")
        .def(py::init<Fragment*,
                      const py::args&,
                      const stelline::BlockShape&,
                      const std::string&>(),
             py::arg("fragment"),
             py::arg("total_block"),
             py::arg("name") = "dummy_receiver");

    py::class_<SorterOp, PySorterOp, Operator, std::shared_ptr<SorterOp>>(m, "SorterOp")
        .def(py::init<Fragment*,
                      const py::args&,
                      uint64_t,
                      const std::string&>(),
             py::arg("fragment"),
             py::arg("depth"),
             py::arg("name") = "sorter")
        .def("tick", &SorterOp::tick)
        .def("format_metrics", &SorterOp::formatMetrics)
        .def("set_manifest_provider", &SorterOp::setManifestProvider)
        .def("set_metrics_provider", &SorterOp::setMetricsProvider)
        .def_property_readonly("manifest", &SorterOp::manifest, py::return_value_policy::reference)
        .def_property_readonly("metrics", &SorterOp::metrics, py::return_value_policy::reference);
}
