#include <matx.h>
#include <holoscan/holoscan.hpp>

#include <stelline/bits/frbnn/base.hh>

using namespace holoscan;
using namespace stelline::bits::frbnn;

class DataGeneratorOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DataGeneratorOp)

    DataGeneratorOp() = default;

    void setup(OperatorSpec& spec) override {
        spec.output<std::shared_ptr<Tensor>>("out");
        spec.param(batch_size, "batch_size", "Batch size.");
        spec.param(test_data_path, "test_data_path", "Path to test data file.");
    }

    void start() {
        // Create data tensor.

        auto tensor = matx::make_tensor<float>({batch_size.get(), 192, 2048});
        data = std::make_shared<Tensor>(tensor.GetDLPackTensor());

        // Load test data from file.

        std::ifstream file(test_data_path.get(), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + test_data_path.get());
        }

        file.read(reinterpret_cast<char*>(data->data()), data->size() * sizeof(float));
        file.close();
    }

    void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
        op_output.emit(data, "out");
    };

 private:
    std::shared_ptr<Tensor> data;
    Parameter<int> batch_size;
    Parameter<std::string> test_data_path;
};

class DummySinkOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DummySinkOp)

    DummySinkOp() = default;

    void setup(OperatorSpec& spec) override {
        spec.input<std::shared_ptr<Tensor>>("in");
        spec.param(batch_size, "batch_size", "Batch size.");
        spec.param(validation_data_path, "validation_data_path", "Path to validation data.");
    }

    void start() {
        // Resize data buffer.

        golden_data.resize(batch_size.get() * 2);
        host_buffer.resize(batch_size.get() * 2);

        // Load golden data from file.

        std::ifstream file(validation_data_path.get(), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + validation_data_path.get());
        }

        file.read(reinterpret_cast<char*>(golden_data.data()), golden_data.size() * sizeof(float));
        file.close();
    }

    void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
        auto tensor = op_input.receive<std::shared_ptr<Tensor>>("in").value();

        // Measure time between messages.

        if (last_time.time_since_epoch().count() != 0) {
            auto now = std::chrono::system_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time);
        }

        // Download tensor data to host and compare with golden data.

        cudaMemcpy(host_buffer.data(), tensor->data(), tensor->size() * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < tensor->size(); i++) {
            if (std::abs(host_buffer[i] - golden_data[i]) > 0.1 || !std::isfinite(host_buffer[i]) || !std::isfinite(golden_data[i])) {
                HOLOSCAN_LOG_ERROR("Mismatch at index {}! Expected: {}, Got: {}", i, golden_data[i], host_buffer[i]);
            }
        }

        // Print statistics.

        HOLOSCAN_LOG_INFO("Model output shape: {}", tensor->shape());
        HOLOSCAN_LOG_INFO("Received message. Took {} ms.", duration.count());

        // Reset timer.

        last_time = std::chrono::system_clock::now();
    };

 private:
    std::chrono::time_point<std::chrono::system_clock> last_time;
    std::chrono::milliseconds duration;
    std::vector<float> golden_data;
    std::vector<float> host_buffer;
    Parameter<uint64_t> batch_size;
    Parameter<std::string> validation_data_path;
};

class FrbnnTestOfflineApp : public holoscan::Application {
 public:
    void compose() override {
        std::shared_ptr<Resource> pool = make_resource<UnboundedAllocator>("pool");

        auto data_gen = make_operator<DataGeneratorOp>("data-gen",
            from_config("test_settings"));
        auto [frbnn_input, frbnn_output] = FrbnnInferenceBit(this, pool, "frbnn_inferencee");
        auto dummy_sink = make_operator<DummySinkOp>("dummy-sink",
            from_config("test_settings"));

        add_flow(data_gen, frbnn_input);
        add_flow(frbnn_output, dummy_sink);
    }
};

int main() {
    auto app = holoscan::make_application<FrbnnTestOfflineApp>();

    app->config(std::filesystem::path("./applications/frbnn/test/offline/default.yaml"));
    app->run();

    return 0;
}
