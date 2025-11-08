#include <stelline/types.hh>
#include <stelline/operators/frbnn/base.hh>

using namespace gxf;
using namespace holoscan;

namespace stelline::operators::frbnn {

//
// ModelPreprocessorOp
//

void ModelPreprocessorOp::setup(OperatorSpec& spec) {
    spec.input<DspBlock>("in");
    spec.output<holoscan::gxf::Entity>("out");
}

void ModelPreprocessorOp::start() {
}

void ModelPreprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
    auto block = input.receive<DspBlock>("in").value();

    const auto& meta = metadata();
    meta->set("dsp_block", block);

    auto outMessage = holoscan::gxf::Entity::New(&context);
    outMessage.add(block.tensor, "input");
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
    spec.output<InferenceBlock>("out");
}

void ModelPostprocessorOp::start() {
}

void ModelPostprocessorOp::compute(InputContext& input, OutputContext& output, ExecutionContext&) {
    auto inputMessage = input.receive<holoscan::gxf::Entity>("in").value();
    auto inputTensor = inputMessage.get<holoscan::Tensor>("output");

    const auto& meta = metadata();

    InferenceBlock block = {
        .dspBlock = meta->get<DspBlock>("dsp_block"),
        .tensor = inputTensor,
    };

    meta->clear();
    output.emit(block, "out");
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

    // Metrics.

    std::thread metricsThread;
    bool metricsThreadRunning;
    void metricsLoop();
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
    spec.input<InferenceBlock>("in")
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

    // Start metrics thread.

    pimpl->metricsThreadRunning = true;
    pimpl->metricsThread = std::thread([&]{
        pimpl->metricsLoop();
    });
}

void SimpleDetectionOp::stop() {
    // Stop metrics thread.

    pimpl->metricsThreadRunning = false;
    if (pimpl->metricsThread.joinable()) {
        pimpl->metricsThread.join();
    }

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
    auto block = input.receive<InferenceBlock>("in").value();

    const auto& outputByteSize = block.tensor->size() * sizeof(float);
    const auto& hitsByteSize = block.dspBlock.tensor->size() * sizeof(float);

    // Download output tensor to host.

    if (pimpl->outputHostBuffer == nullptr) {
        if (cudaMallocHost(&pimpl->outputHostBuffer, outputByteSize) != cudaSuccess) {
            HOLOSCAN_LOG_ERROR("Failed to allocate pinned host output buffer.");
            throw std::runtime_error("Failed to write to file.");
        }
    }

    cudaMemcpy(pimpl->outputHostBuffer, block.tensor->data(), outputByteSize, cudaMemcpyDeviceToHost);

    // Detect hits, log statistics, and write hits to file.

    for (int i = 0; i < block.tensor->shape()[0]; i++) {
        const float* val = reinterpret_cast<const float*>(pimpl->outputHostBuffer);
        const auto& a = val[i * 2];
        const auto& b = val[i * 2 + 1];

        const auto p1 = std::chrono::system_clock::now();
        const auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        std::string csvLine = fmt::format("{},{},{},{},{},{},{},{}\n", pimpl->iterations++,
                                                                       a,
                                                                       b,
                                                                       (a > b) ? 0 : 1,
                                                                       (a > b) ? "NO" : ">> HIT <<",
                                                                       i,
                                                                       block.dspBlock.timestamp,
                                                                       timestamp);
        pimpl->csvFile.write(csvLine.c_str(), csvLine.size());
        pimpl->csvFile.flush();

        if (a < b) {
            std::ofstream hitsFile;
            std::string hitsFilePath = fmt::format("{}/FRBNN-HIT-{}.bin", pimpl->hitsDirectory, block.dspBlock.timestamp);
            hitsFile.open(hitsFilePath, std::ios::out | std::ios::binary);

            if (pimpl->hitsHostBuffer == nullptr) {
                if (cudaMallocHost(&pimpl->hitsHostBuffer, hitsByteSize) != cudaSuccess) {
                    HOLOSCAN_LOG_ERROR("Failed to allocate pinned host hits buffer.");
                    throw std::runtime_error("Failed to write to file.");
                }
            }

            cudaMemcpyAsync(pimpl->hitsHostBuffer, block.dspBlock.tensor->data(), hitsByteSize, cudaMemcpyDeviceToHost, pimpl->stream);
            cudaStreamSynchronize(pimpl->stream);

            hitsFile.write(reinterpret_cast<char*>(pimpl->hitsHostBuffer), hitsByteSize);
            hitsFile.flush();
            hitsFile.close();

            pimpl->numberOfHits++;
        }
    }
}

void SimpleDetectionOp::Impl::metricsLoop() {
    while (metricsThreadRunning) {
        HOLOSCAN_LOG_INFO("Simple Detector Operator:");
        HOLOSCAN_LOG_INFO("  Iterations: {}", iterations);
        HOLOSCAN_LOG_INFO("  Hits      : {}", numberOfHits);

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

}  // namespace stelline::operators::frbnn
