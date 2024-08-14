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
        spec.param(batchSize, "batch_size", "Batch size.");
        spec.param(testDataPath, "test_data_path", "Path to test data file.");
    }

    void start() {
        // Create data tensor.

        auto tensor = matx::make_tensor<float>({batchSize.get(), 192, 2048});
        data = std::make_shared<Tensor>(tensor.GetDLPackTensor());

        // Load test data from file.

        std::ifstream file(testDataPath.get(), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + testDataPath.get());
        }

        file.read(reinterpret_cast<char*>(data->data()), data->size() * sizeof(float));
        file.close();
    }

    void compute(InputContext& input, OutputContext& output, ExecutionContext&) override {
        output.emit(data, "out");
    };

 private:
    std::shared_ptr<Tensor> data;
    Parameter<int> batchSize;
    Parameter<std::string> testDataPath;
};

class DummySinkOp : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(DummySinkOp)

    DummySinkOp() = default;

    void setup(OperatorSpec& spec) override {
        spec.input<std::shared_ptr<Tensor>>("in");
        spec.param(batchSize, "batch_size", "Batch size.");
        spec.param(validationDataPath, "validation_data_path", "Path to validation data.");
    }

    void start() {
        // Resize data buffer.

        goldenData.resize(batchSize.get() * 2);
        hostBuffer.resize(batchSize.get() * 2);

        // Load golden data from file.

        std::ifstream file(validationDataPath.get(), std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + validationDataPath.get());
        }

        file.read(reinterpret_cast<char*>(goldenData.data()), goldenData.size() * sizeof(float));
        file.close();
    }

    void compute(InputContext& input, OutputContext&, ExecutionContext&) override {
        auto tensor = input.receive<std::shared_ptr<Tensor>>("in").value();

        // Measure time between messages.

        if (lastTime.time_since_epoch().count() != 0) {
            auto now = std::chrono::system_clock::now();
            duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastTime);
        }

        // Download tensor data to host and compare with golden data.

        cudaMemcpy(hostBuffer.data(), tensor->data(), tensor->size() * sizeof(float), cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < tensor->size(); i++) {
            if (std::abs(hostBuffer[i] - goldenData[i]) > 0.1 || !std::isfinite(hostBuffer[i]) || !std::isfinite(goldenData[i])) {
                HOLOSCAN_LOG_ERROR("Mismatch at index {}! Expected: {}, Got: {}", i, goldenData[i], hostBuffer[i]);
            }
        }

        // Print statistics.

        HOLOSCAN_LOG_INFO("Model output shape: {}", tensor->shape());
        HOLOSCAN_LOG_INFO("Received message. Took {} ms.", duration.count());

        // Reset timer.

        lastTime = std::chrono::system_clock::now();
    };

 private:
    std::chrono::time_point<std::chrono::system_clock> lastTime;
    std::chrono::milliseconds duration;
    std::vector<float> goldenData;
    std::vector<float> hostBuffer;
    Parameter<uint64_t> batchSize;
    Parameter<std::string> validationDataPath;
};

class FrbnnTestOfflineApp : public holoscan::Application {
 public:
    void compose() override {
        std::shared_ptr<Resource> pool = make_resource<UnboundedAllocator>("pool");

        auto dataGen = make_operator<DataGeneratorOp>("data-gen",
            from_config("test_settings"));
        auto [frbnnInput, frbnnOutput] = FrbnnInferenceBit(this, pool, "frbnn_inference");
        auto dummySink = make_operator<DummySinkOp>("dummy-sink",
            from_config("test_settings"));

        add_flow(dataGen, frbnnInput);
        add_flow(frbnnOutput, dummySink);
    }
};

int main() {
    auto app = holoscan::make_application<FrbnnTestOfflineApp>();

    app->config(std::filesystem::path("./applications/frbnn/test/offline/default.yaml"));
    app->run();

    return 0;
}
