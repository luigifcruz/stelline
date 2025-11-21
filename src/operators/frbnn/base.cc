#include <stelline/types.hh>
#include <stelline/operators/frbnn/base.hh>
#include <fmt/format.h>

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::frbnn {

//
// ModelPreprocessorOp
//

void ModelPreprocessorOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelPreprocessorOp::start() {
}

void ModelPreprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
    auto block = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();

    const auto& meta = metadata();
    meta->set("dsp_block", block);

    auto outMessage = holoscan::gxf::Entity::New(&context);
    outMessage.add(block, "input");
    output.emit(outMessage, "out");
};

//
// ModelAdapterOp
//

void ModelAdapterOp::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelAdapterOp::start() {
}

void ModelAdapterOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
    auto inputMessage = input.receive<holoscan::gxf::Entity>("in").value();
    auto inputTensor = inputMessage.get<holoscan::Tensor>("output");

    auto outMessage = holoscan::gxf::Entity::New(&context);
    outMessage.add(inputTensor, "input");
    output.emit(outMessage, "out");
};

//
// ModelPostprocessorOp
//

void ModelPostprocessorOp::setup(OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<std::shared_ptr<holoscan::Tensor>>("out");
}

void ModelPostprocessorOp::start() {
}

void ModelPostprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto inputMessage = input.receive<holoscan::gxf::Entity>("in").value();
    auto inputTensor = inputMessage.get<holoscan::Tensor>("output");

    output.emit(inputTensor, "out");
};

//
// SimpleDetectionOp
//

struct SimpleDetectionOp::Impl {
    // State.

    uint64_t numberOfHits;
    uint64_t iterations;
    cudaStream_t stream;
    std::ofstream csvFile;
    std::string csvFilePath;
    std::string hitsDirectory;
    void* outputHostBuffer;
    void* hitsHostBuffer;
};

void SimpleDetectionOp::initialize() {
    // Allocate memory.
    pimpl = new Impl();

    // Initialize operator.
    Operator::initialize();
}

SimpleDetectionOp::~SimpleDetectionOp() {
    delete pimpl;
}

void SimpleDetectionOp::setup(OperatorSpec& spec) {
    spec.input<std::shared_ptr<holoscan::Tensor>>("in")
        .connector(IOSpec::ConnectorType::kDoubleBuffer,
                   holoscan::Arg("capacity", 1024UL));

    spec.param(csvFilePath_, "csv_file_path");
    spec.param(hitsDirectory_, "hits_directory");
}

void SimpleDetectionOp::start() {
    // Convert Parameters to variables.

    pimpl->csvFilePath = csvFilePath_.get();
    pimpl->hitsDirectory = hitsDirectory_.get();

    // Open and initialize CSV file.

    pimpl->csvFile.open(pimpl->csvFilePath, std::ios::out);

    std::string header = fmt::format("Iteration,Result A,Result B,Argmax,Hit,Batch Index,Telescope Timestamp,UNIX Timestamp\n");
    pimpl->csvFile.write(header.c_str(), header.size());
    pimpl->csvFile.flush();

    // Create stream.
    cudaStreamCreateWithFlags(&pimpl->stream, cudaStreamNonBlocking);
}

void SimpleDetectionOp::stop() {
    // Free host buffers.

    if (pimpl->outputHostBuffer != nullptr) {
        cudaFreeHost(pimpl->outputHostBuffer);
    }

    if (pimpl->hitsHostBuffer != nullptr) {
        cudaFreeHost(pimpl->hitsHostBuffer);
    }

    // Destroy CUDA stream.
    cudaStreamDestroy(pimpl->stream);

    // Close CSV file.
    pimpl->csvFile.close();
}

void SimpleDetectionOp::compute(InputContext& input, OutputContext&, ExecutionContext&) {
    const auto& meta = metadata();

    const auto& inferenceTensor = input.receive<std::shared_ptr<holoscan::Tensor>>("in").value();
    const auto& originalTensor = meta->get<std::shared_ptr<holoscan::Tensor>>("dsp_block");
    const auto& timestamp = meta->get<uint64_t>("timestamp");

    const auto& outputByteSize = inferenceTensor->size() * sizeof(float);
    const auto& hitsByteSize = originalTensor->size() * sizeof(float);

    // Download output tensor to host.

    if (pimpl->outputHostBuffer == nullptr) {
        if (cudaMallocHost(&pimpl->outputHostBuffer, outputByteSize) != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to allocate pinned host output buffer.");
            throw std::runtime_error("Failed to write to file.");
        }
    }

    cudaMemcpy(pimpl->outputHostBuffer, inferenceTensor->data(), outputByteSize, cudaMemcpyDeviceToHost);

    // Detect hits, log statistics, and write hits to file.

    for (int i = 0; i < inferenceTensor->shape()[0]; i++) {
        const float* val = reinterpret_cast<const float*>(pimpl->outputHostBuffer);
        const auto& a = val[i * 2];
        const auto& b = val[i * 2 + 1];

        const auto p1 = std::chrono::system_clock::now();
        const auto systemTimestamp = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        std::string csvLine = fmt::format("{},{},{},{},{},{},{},{}\n", pimpl->iterations++,
                                                                       a,
                                                                       b,
                                                                       (a > b) ? 0 : 1,
                                                                       (a > b) ? "NO" : ">> HIT <<",
                                                                       i,
                                                                       timestamp,
                                                                       systemTimestamp);
        pimpl->csvFile.write(csvLine.c_str(), csvLine.size());
        pimpl->csvFile.flush();

        if (a < b) {
            std::ofstream hitsFile;
            std::string hitsFilePath = fmt::format("{}/FRBNN-HIT-{}.bin", pimpl->hitsDirectory, timestamp);
            hitsFile.open(hitsFilePath, std::ios::out | std::ios::binary);

            if (pimpl->hitsHostBuffer == nullptr) {
                if (cudaMallocHost(&pimpl->hitsHostBuffer, hitsByteSize) != cudaSuccess) {
                    HOLOSCAN_LOG_ERROR("Failed to allocate pinned host hits buffer.");
                    throw std::runtime_error("Failed to write to file.");
                }
            }

            cudaMemcpyAsync(pimpl->hitsHostBuffer, originalTensor->data(), hitsByteSize, cudaMemcpyDeviceToHost, pimpl->stream);
            cudaStreamSynchronize(pimpl->stream);

            hitsFile.write(reinterpret_cast<char*>(pimpl->hitsHostBuffer), hitsByteSize);
            hitsFile.flush();
            hitsFile.close();

            pimpl->numberOfHits++;
        }
    }
}

stelline::StoreInterface::MetricsMap SimpleDetectionOp::collectMetricsMap() {
    if (!pimpl) {
        return {};
    }
    stelline::StoreInterface::MetricsMap metrics;
    metrics["iterations"] = fmt::format("{}", pimpl->iterations);
    metrics["hits"] = fmt::format("{}", pimpl->numberOfHits);
    return metrics;
}

std::string SimpleDetectionOp::collectMetricsString() {
    if (!pimpl) {
        return {};
    }
    const auto metrics = collectMetricsMap();
    return fmt::format("  Iterations: {}\n"
                       "  Hits      : {}",
                       metrics.at("iterations"),
                       metrics.at("hits"));
}

}  // namespace stelline::operators::frbnn
