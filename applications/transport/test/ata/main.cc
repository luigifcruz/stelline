#include <holoscan/holoscan.hpp>

#include <stelline/bits/transport/base.hh>

using namespace holoscan;
using namespace stelline::bits::transport;

class DummySinkOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DummySinkOp)

    DummySinkOp() = default;

    void setup(OperatorSpec& spec) override {
        spec.input<std::shared_ptr<Tensor>>("in").connector(IOSpec::ConnectorType::kDoubleBuffer, holoscan::Arg("capacity", 32UL));
    }

    void compute(InputContext& input, OutputContext&, ExecutionContext&) override {
        auto tensor = input.receive<std::shared_ptr<Tensor>>("in").value();

        // Measure time between messages.

        if (lastTime.time_since_epoch().count() != 0) {
            auto now = std::chrono::system_clock::now();
            duration += std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime);
        }

        // Print statistics.

        if (numIterations++ % 100 == 0) {
            HOLOSCAN_LOG_INFO("Model output shape: {}", tensor->shape());
            HOLOSCAN_LOG_INFO("Received message. Took {} ms.", duration.count() / 100);
            duration = std::chrono::milliseconds(0);
        }

        // Reset timer.

        lastTime = std::chrono::system_clock::now();
    };

 private:
    std::chrono::time_point<std::chrono::system_clock> lastTime;
    std::chrono::milliseconds duration;
    uint64_t numIterations;
};

class TransportTestApp : public holoscan::Application {
 public:
    void compose() override {
        std::shared_ptr<Resource> pool = make_resource<UnboundedAllocator>("pool");

        auto transportOutput = TransportBit(this, pool, "ata_standard_rx");
        auto dummySink = make_operator<DummySinkOp>("dummy-sink");

        add_flow(transportOutput, dummySink, {{"dsp_block_out", "in"}});
    }
};

int main() {
    auto app = holoscan::make_application<TransportTestApp>();

    app->config(std::filesystem::path("./applications/transport/test/ata/default.yaml"));
    app->run();

    return 0;
}
