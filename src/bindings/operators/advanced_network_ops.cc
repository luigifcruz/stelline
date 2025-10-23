#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/operators/advanced_network/adv_network_rx.h>
#include <holoscan/operators/advanced_network/adv_network_common.h>

#include <string>
#include <vector>

namespace py = pybind11;
using namespace holoscan;
using namespace holoscan::ops;
using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

class PyAdvNetworkOpRx : public ops::AdvNetworkOpRx {
public:
    using ops::AdvNetworkOpRx::AdvNetworkOpRx;

    PyAdvNetworkOpRx(Fragment* fragment,
                     int master_core,
                     const std::string& interface_name,
                     int gpu_device,
                     const std::vector<int>& worker_cores,
                     int max_packet_size,
                     int num_concurrent_batches,
                     int batch_size,
                     int header_data_split,
                     int udp_src_port,
                     int udp_dst_port,
                     const std::string& output_port_name = "bench_rx_out",
                     const std::string& queue_name = "adc_rx",
                     const std::string& flow_name = "adc_rx",
                     const std::string& name = "advanced_network_rx") {

        ops::AdvNetConfigYaml ano_cfg = {};
        ano_cfg.common_.version = 1;
        ano_cfg.common_.master_core_ = master_core;
        ano_cfg.common_.dir = ops::AdvNetDirection::RX;

        ops::AdvNetRxConfig ano_cfg_interface = {};
        ano_cfg_interface.if_name_ = interface_name;
        ano_cfg_interface.port_id_ = 0;
        ano_cfg_interface.flow_isolation_ = true;

        ops::RxQueueConfig ano_cfg_rx_queue = {};
        ano_cfg_rx_queue.common_.name_ = queue_name;
        ano_cfg_rx_queue.common_.id_ = 0;
        ano_cfg_rx_queue.common_.gpu_dev_ = gpu_device;
        ano_cfg_rx_queue.common_.gpu_direct_ = true;
        ano_cfg_rx_queue.common_.hds_ = header_data_split;

        std::string cpu_cores_str;
        for (size_t i = 0; i < worker_cores.size(); ++i) {
            if (i > 0) cpu_cores_str += ",";
            cpu_cores_str += std::to_string(worker_cores[i]);
        }
        ano_cfg_rx_queue.common_.cpu_cores_ = cpu_cores_str;

        ano_cfg_rx_queue.common_.max_packet_size_ = max_packet_size;
        ano_cfg_rx_queue.common_.num_concurrent_batches_ = num_concurrent_batches;
        ano_cfg_rx_queue.common_.batch_size_ = batch_size;
        ano_cfg_rx_queue.output_port_ = output_port_name;

        ano_cfg_interface.queues_.push_back(ano_cfg_rx_queue);

        ops::FlowConfig ano_cfg_flow = {};
        ano_cfg_flow.name_ = flow_name;
        ano_cfg_flow.action_.type_ = ops::FlowType::QUEUE;
        ano_cfg_flow.action_.id_ = 0;
        ano_cfg_flow.match_.udp_src_ = udp_src_port;
        ano_cfg_flow.match_.udp_dst_ = udp_dst_port;

        ano_cfg_interface.flows_.push_back(ano_cfg_flow);

        ano_cfg.rx_.push_back(ano_cfg_interface);

        this->add_arg(Arg("cfg", ano_cfg));

        name_ = name;
        fragment_ = fragment;
        spec_ = std::make_shared<OperatorSpec>(fragment);
        setup(*spec_.get());
    }
};

PYBIND11_MODULE(_advanced_network_ops, m) {
    m.doc() = "Stelline advanced network operators module";

    py::class_<ops::AdvNetworkOpRx, PyAdvNetworkOpRx, Operator, std::shared_ptr<ops::AdvNetworkOpRx>>(
        m, "AdvNetworkOpRx")
        .def(py::init<Fragment*,
                      int,
                      const std::string&,
                      int,
                      const std::vector<int>&,
                      int,
                      int,
                      int,
                      int,
                      int,
                      int,
                      const std::string&,
                      const std::string&,
                      const std::string&,
                      const std::string&>(),
             "fragment"_a,
             "master_core"_a,
             "interface_name"_a,
             "gpu_device"_a,
             "worker_cores"_a,
             "max_packet_size"_a,
             "num_concurrent_batches"_a,
             "batch_size"_a,
             "header_data_split"_a,
             "udp_src_port"_a,
             "udp_dst_port"_a,
             "output_port_name"_a = "bench_rx_out"s,
             "queue_name"_a = "adc_rx"s,
             "flow_name"_a = "adc_rx"s,
             "name"_a = "advanced_network_rx"s)
        .def("initialize", &ops::AdvNetworkOpRx::initialize)
        .def("setup", &ops::AdvNetworkOpRx::setup, "spec"_a);
}
