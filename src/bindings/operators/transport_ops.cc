#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <holoscan/core/application.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>

#include <stelline/operators/transport/base.hh>
#include <stelline/yaml/types/block_shape.hh>

#include <advanced_network/types.h>
#include <advanced_network/common.h>

namespace py = pybind11;
using namespace stelline::operators::transport;
using namespace holoscan;

class PyAtaReceiverOp : public AtaReceiverOp {
public:
    using AtaReceiverOp::AtaReceiverOp;

    PyAtaReceiverOp(Fragment* fragment,
                    const py::args& args,
                    int gpu_device_id,
                    const std::string& interface_address,
                    int master_core,
                    int worker_core,
                    const std::string& packet_parser_type,
                    uint64_t packet_header_offset,
                    uint64_t packet_header_size,
                    int packet_data_size,
                    uint64_t packets_per_burst,
                    uint64_t max_concurrent_bursts,
                    uint64_t max_concurrent_blocks,
                    int udp_src_port,
                    int udp_dst_port,
                    const stelline::BlockShape& total_block,
                    const stelline::BlockShape& partial_block,
                    const stelline::BlockShape& offset_block,
                    uint64_t output_pool_size,
                    bool enable_csv_logging,
                    const std::string& name = "ata_receiver")
        : AtaReceiverOp(ArgList{Arg("max_concurrent_blocks", max_concurrent_blocks),
                                Arg("total_block", total_block),
                                Arg("packet_header_size", packet_header_size),
                                Arg("packet_header_offset", packet_header_offset),
                                Arg("partial_block", partial_block),
                                Arg("offset_block", offset_block),
                                Arg("output_pool_size", output_pool_size),
                                Arg("enable_csv_logging", enable_csv_logging)}) {
        // Configure ANO.

        {
            using namespace holoscan::advanced_network;

            NetworkConfig cfg = {};

            cfg.log_level_ =  holoscan::advanced_network::LogLevel::TRACE;
            cfg.tx_meta_buffers_ = DEFAULT_TX_META_BUFFERS;
            cfg.rx_meta_buffers_ = DEFAULT_RX_META_BUFFERS;

            cfg.common_.version = 1;
            cfg.common_.master_core_ = master_core;
            cfg.common_.dir = Direction::RX;
            cfg.common_.manager_type = ManagerType::DPDK;
            cfg.common_.loopback_ = LoopbackType::DISABLED;

            {
                MemoryRegionConfig memory_cfg = {};

                memory_cfg.name_ = "RX_HEADER";
                memory_cfg.kind_ = MemoryKind::HUGE;
                memory_cfg.affinity_ = 0;
                memory_cfg.buf_size_ = packet_header_offset + packet_header_size;
                memory_cfg.num_bufs_ = packets_per_burst * max_concurrent_bursts;
                memory_cfg.access_ = MEM_ACCESS_LOCAL;
                memory_cfg.owned_ = true;

                cfg.mrs_.emplace(memory_cfg.name_, memory_cfg);
            }

            {
                MemoryRegionConfig memory_cfg = {};

                memory_cfg.name_ = "RX_DATA";
                memory_cfg.kind_ = MemoryKind::HOST_PINNED;
                memory_cfg.affinity_ = 0;
                memory_cfg.buf_size_ = packet_data_size;
                memory_cfg.num_bufs_ = packets_per_burst * max_concurrent_bursts;
                memory_cfg.access_ = MEM_ACCESS_LOCAL;
                memory_cfg.owned_ = true;

                cfg.mrs_.emplace(memory_cfg.name_, memory_cfg);
            }

            {
                InterfaceConfig interface_cfg = {};

                interface_cfg.address_ = interface_address;
                interface_cfg.rx_.flow_isolation_ = false;

                {
                    RxQueueConfig queue_cfg = {};

                    queue_cfg.common_.name_ = "main";
                    queue_cfg.common_.id_ = 0;
                    queue_cfg.common_.batch_size_ = packets_per_burst;
                    queue_cfg.common_.cpu_core_ = fmt::format("{}", worker_core);
                    queue_cfg.common_.mrs_ = {"RX_HEADER", "RX_DATA"};
                    queue_cfg.timeout_us_ = 0;

                    interface_cfg.rx_.queues_.push_back(queue_cfg);
                }

                {
                    FlowConfig flow_cfg = {};

                    flow_cfg.name_ = "main";
                    flow_cfg.id_ = 0;
                    flow_cfg.action_.type_ = FlowType::QUEUE;
                    flow_cfg.action_.id_ = 0;
                    flow_cfg.match_.udp_src_ = udp_src_port;
                    flow_cfg.match_.udp_dst_ = udp_dst_port;
                    flow_cfg.match_.ipv4_len_ = 0;

                    interface_cfg.rx_.flows_.push_back(flow_cfg);
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
                      int,
                      const std::string&,
                      int,
                      int,
                      int,
                      uint64_t,
                      uint64_t,
                      uint64_t,
                      int,
                      int,
                      const stelline::BlockShape&,
                      const stelline::BlockShape&,
                      const stelline::BlockShape&,
                      uint64_t,
                      bool,
                      const std::string&>(),
             py::arg("fragment"),
             py::arg("gpu_device_id"),
             py::arg("interface_address"),
             py::arg("master_core"),
             py::arg("worker_core"),
             py::arg("packet_parser_type"),
             py::arg("packet_header_offset"),
             py::arg("packet_header_size"),
             py::arg("packet_data_size"),
             py::arg("packets_per_burst"),
             py::arg("max_concurrent_bursts"),
             py::arg("max_concurrent_blocks"),
             py::arg("udp_src_port"),
             py::arg("udp_dst_port"),
             py::arg("total_block"),
             py::arg("partial_block"),
             py::arg("offset_block"),
             py::arg("output_pool_size"),
             py::arg("enable_csv_logging"),
             py::arg("name") = "ata_receiver")
        .def("collect_metrics_map", &AtaReceiverOp::collectMetricsMap)
        .def("collect_metrics_string", &AtaReceiverOp::collectMetricsString);

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
        .def("collect_metrics_map", &SorterOp::collectMetricsMap)
        .def("collect_metrics_string", &SorterOp::collectMetricsString);
}
