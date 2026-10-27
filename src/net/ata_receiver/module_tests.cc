#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <jetstream/detail/scheduler_impl.hh>
#include <jetstream/module_context.hh>
#include <jetstream/module_interface.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>

#include <stelline/ata_receiver/module.hh>

#include "detail/daqiri_config.hh"
#include "detail/kernel.hh"
#include "module_impl.hh"

using namespace Jetstream;

namespace {

struct AtaReceiverValidationImpl : Modules::AtaReceiverImpl {
    using AtaReceiverImpl::validatedPacketSizeBytes;
    using AtaReceiverImpl::validatedSubscriptions;
};

struct AtaReceiverReadinessImpl : Modules::AtaReceiverImpl {
    using AtaReceiverImpl::packetProcessingThreadRunning;
    using AtaReceiverImpl::popReadyTensor;
    using AtaReceiverImpl::pushReadyTensor;
    using AtaReceiverImpl::reportThreadFailure;
};

Modules::AtaReceiver ValidConfig() {
    Modules::AtaReceiver config;
    config.interfaceAddress = "lo";
    config.workerCores = {0};
    config.subscriptions = "- 127.0.0.1:10000 -> 239.1.1.1:50000";
    config.totalBlock = {1, 96, 16, 2};
    config.partialBlock = {1, 96, 16, 2};
    config.packetsPerBurst = 1;
    config.maxConcurrentBursts = 1;
    return config;
}

std::shared_ptr<Module> BuildModule() {
    const auto implementations = Registry::ListAvailableModules("ata_receiver");
    REQUIRE(implementations.size() == 1);

    const auto& implementation = implementations.front();
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("ata_receiver",
                                  implementation.device,
                                  implementation.runtime,
                                  implementation.provider,
                                  module) == Result::SUCCESS);
    return module;
}

struct ScatterBuffers {
    void* input = nullptr;
    void* output = nullptr;
    const void** sources = nullptr;
    void** destinations = nullptr;
    cudaStream_t stream = nullptr;

    ~ScatterBuffers() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
        cudaFree(input);
        cudaFree(output);
        cudaFreeHost(sources);
        cudaFreeHost(destinations);
    }
};

template<typename Scalar>
void CheckScatter(const Modules::AtaReceiverBlockGeometry& partial,
                  const U64 totalPolarizations,
                  const std::string& outputType) {
    const Modules::AtaReceiverBlockGeometry total{
        partial.numberOfAntennas * 2,
        partial.numberOfChannels * 2,
        partial.numberOfSamples * 2,
        totalPolarizations,
    };
    constexpr U64 packets = 3;
    constexpr U64 guardScalars = 32;
    const U64 packetScalars = partial.numberOfAntennas * partial.numberOfChannels *
                              partial.numberOfSamples * partial.numberOfPolarizations * 2;
    const U64 outputScalars = total.numberOfAntennas * total.numberOfChannels *
                              total.numberOfSamples * total.numberOfPolarizations * 2;
    const std::array<std::array<U64, 4>, packets> origins{{
        {0, 0, 0, 0},
        {partial.numberOfAntennas, partial.numberOfChannels, partial.numberOfSamples, 0},
        {0, partial.numberOfChannels, 0, totalPolarizations - partial.numberOfPolarizations},
    }};
    const auto offset = [&](U64 a, U64 c, U64 s, U64 p) {
        return (((a * total.numberOfChannels + c) * total.numberOfSamples + s) *
                total.numberOfPolarizations + p) * 2;
    };

    std::vector<std::int8_t> input(packets * packetScalars);
    for (U64 i = 0; i < input.size(); ++i) {
        input[i] = static_cast<std::int8_t>(static_cast<int>((i * 73 + i / 256) % 256) - 128);
    }
    std::vector<Scalar> output(outputScalars + 2 * guardScalars, Scalar{37});
    auto expected = output;
    for (U64 packet = 0; packet < packets; ++packet) {
        const auto& origin = origins[packet];
        U64 source = packet * packetScalars;
        for (U64 a = 0; a < partial.numberOfAntennas; ++a) {
            for (U64 c = 0; c < partial.numberOfChannels; ++c) {
                for (U64 s = 0; s < partial.numberOfSamples; ++s) {
                    for (U64 p = 0; p < partial.numberOfPolarizations; ++p) {
                        const U64 destination = guardScalars +
                            offset(origin[0] + a, origin[1] + c, origin[2] + s, origin[3] + p);
                        expected[destination] = static_cast<Scalar>(input[source++]);
                        expected[destination + 1] = static_cast<Scalar>(input[source++]);
                    }
                }
            }
        }
    }

    ScatterBuffers buffers;
    REQUIRE(cudaStreamCreateWithFlags(&buffers.stream, cudaStreamNonBlocking) == cudaSuccess);
    REQUIRE(cudaMalloc(&buffers.input, input.size()) == cudaSuccess);
    REQUIRE(cudaMalloc(&buffers.output, output.size() * sizeof(Scalar)) == cudaSuccess);
    // Match production's pinned, host-resident pointer tables.
    REQUIRE(cudaMallocHost(&buffers.sources, packets * sizeof(void*)) == cudaSuccess);
    REQUIRE(cudaMallocHost(&buffers.destinations, packets * sizeof(void*)) == cudaSuccess);
    REQUIRE(cudaMemcpy(buffers.input, input.data(), input.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(buffers.output, output.data(), output.size() * sizeof(Scalar),
                       cudaMemcpyHostToDevice) == cudaSuccess);
    for (U64 i = 0; i < packets; ++i) {
        const U64 packet = (i + 2) % packets;
        const auto& origin = origins[packet];
        buffers.sources[i] = static_cast<std::int8_t*>(buffers.input) + packet * packetScalars;
        buffers.destinations[i] = static_cast<Scalar*>(buffers.output) + guardScalars +
            offset(origin[0], origin[1], origin[2], origin[3]);
    }
    REQUIRE(Modules::LaunchAtaScatterKernel(buffers.sources, buffers.destinations, packets,
                                            total, partial, outputType, buffers.stream) == cudaSuccess);
    REQUIRE(cudaStreamSynchronize(buffers.stream) == cudaSuccess);
    REQUIRE(cudaMemcpy(output.data(), buffers.output, output.size() * sizeof(Scalar),
                       cudaMemcpyDeviceToHost) == cudaSuccess);
    REQUIRE(output == expected);
}

}  // namespace

TEST_CASE("ATA CUDA scatter preserves packet samples and untouched tensor regions",
          "[stelline][ata_receiver][module][cuda][scatter]") {
    int deviceCount = 0;
    const auto status = cudaGetDeviceCount(&deviceCount);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && deviceCount == 0)) {
        SKIP("CUDA scatter requires an available CUDA device.");
    }
    REQUIRE(status == cudaSuccess);

    const auto [partial, totalPolarizations] = GENERATE(
        Catch::Generators::table<Modules::AtaReceiverBlockGeometry, U64>({
            {{1, 96, 16, 2}, 2},
            {{1, 128, 16, 2}, 2},
            {{1, 1, 512, 2}, 2},
            {{1, 1, 1, 2}, 2},
            {{2, 3, 5, 1}, 2},
            {{2, 7, 19, 2}, 2},
        }));
    const std::string outputType = GENERATE("CI8", "CF32");
    CAPTURE(outputType, partial.numberOfAntennas, partial.numberOfChannels,
            partial.numberOfSamples, partial.numberOfPolarizations, totalPolarizations);
    if (outputType == "CI8") {
        CheckScatter<std::int8_t>(partial, totalPolarizations, outputType);
    } else {
        CheckScatter<float>(partial, totalPolarizations, outputType);
    }
}

TEST_CASE("Registered ATA CUDA runtime uses nonblocking receiver readiness",
          "[stelline][ata_receiver][module][scheduler][cuda]") {
    auto module = BuildModule();
    REQUIRE(module->device() == DeviceType::CUDA);

    // Exercise production virtual dispatch without opening a NIC or allocating GPU resources.
    auto* impl = module->getImpl<Modules::AtaReceiverImpl>();
    REQUIRE(impl != nullptr);
    auto& receiving = impl->*&AtaReceiverReadinessImpl::packetProcessingThreadRunning;
    receiving.store(true);

    // A stopped receiver could bypass the old wait. Keep reception active and the queue empty.
    const auto start = std::chrono::steady_clock::now();
    for (int attempt = 0; attempt < 8; ++attempt) {
        REQUIRE(module->context()->scheduler()->hasPendingCompute() == Result::TIMEOUT);
    }
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500));

    const auto runtime = std::dynamic_pointer_cast<NativeCudaRuntimeContext>(
        module->context()->runtime());
    REQUIRE(runtime != nullptr);
    REQUIRE(runtime->computeSubmit(nullptr) == Result::TIMEOUT);
    receiving.store(false);
}

TEST_CASE("ATA receiver readiness is nonblocking and preserves queued frames",
          "[stelline][ata_receiver][module][scheduler]") {
    AtaReceiverReadinessImpl impl;
    impl.packetProcessingThreadRunning.store(true);

    const auto start = std::chrono::steady_clock::now();
    for (int attempt = 0; attempt < 8; ++attempt) {
        REQUIRE(impl.hasPendingCompute() == Result::TIMEOUT);
    }
    REQUIRE(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(500));

    const auto first = std::make_shared<Tensor>();
    const auto second = std::make_shared<Tensor>();
    const U64 epoch = Scheduler::Impl::BeginComputeRound();
    REQUIRE(impl.pushReadyTensor(first, 123) == Result::SUCCESS);
    REQUIRE(Scheduler::Impl::BeginComputeRound() != epoch);
    REQUIRE(impl.pushReadyTensor(second, 456) == Result::SUCCESS);

    REQUIRE(impl.hasPendingCompute() == Result::SUCCESS);
    REQUIRE(impl.hasPendingCompute() == Result::SUCCESS);
    REQUIRE(impl.getReadyQueue() == 2);

    Modules::AtaReceiverReadyTensor ready;
    REQUIRE(impl.popReadyTensor(ready) == Result::SUCCESS);
    REQUIRE(ready.tensor == first);
    REQUIRE(ready.timestamp == 123);
    REQUIRE(impl.hasPendingCompute() == Result::SUCCESS);

    REQUIRE(impl.popReadyTensor(ready) == Result::SUCCESS);
    REQUIRE(ready.tensor == second);
    REQUIRE(ready.timestamp == 456);
    REQUIRE(impl.hasPendingCompute() == Result::TIMEOUT);
    REQUIRE(impl.popReadyTensor(ready) == Result::YIELD);
}

TEST_CASE("ATA receiver wakes the scheduler and retains background failure diagnostics",
          "[stelline][ata_receiver][module][scheduler][diagnostic]") {
    AtaReceiverReadinessImpl impl;
    REQUIRE(impl.pushReadyTensor(std::make_shared<Tensor>(), 123) == Result::SUCCESS);

    const U64 epoch = Scheduler::Impl::BeginComputeRound();
    impl.reportThreadFailure("Receiver failed while collecting packets.");
    REQUIRE(Scheduler::Impl::BeginComputeRound() != epoch);
    REQUIRE(impl.hasPendingCompute() == Result::ERROR);
    REQUIRE(JST_LOG_LAST_ERROR() == "Receiver failed while collecting packets.");
    REQUIRE(impl.getReadyQueue() == 1);
}

TEST_CASE("ATA receiver validates bounded plans before defining its interface",
          "[stelline][ata_receiver][module][validation][lifecycle]") {
    auto module = BuildModule();

    auto config = ValidConfig();
    config.totalBlock = {
        std::numeric_limits<U64>::max(), 96, 16, 2,
    };

    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->interface()->outputs().empty());
    REQUIRE(module->outputs().empty());
}

TEST_CASE("ATA receiver sizes packet buffers from the fragment shape",
          "[stelline][ata_receiver][module][validation]") {
    const auto [partialBlock, expectedPayloadSize] = GENERATE(Catch::Generators::table<Shape, U64>({
        {Shape{1, 1, 512, 2}, 2048},
        {Shape{1, 96, 16, 2}, 6144},
        {Shape{1, 128, 16, 2}, 8192},
    }));
    AtaReceiverValidationImpl impl;
    auto& config = *impl.candidate();
    config = ValidConfig();
    config.partialBlock = partialBlock;
    config.totalBlock = {2, partialBlock[1] * 2, partialBlock[2] * 2, 2};
    config.dataType = GENERATE("CI8", "CF32");

    REQUIRE(impl.validate() == Result::SUCCESS);
    REQUIRE(impl.validatedPacketSizeBytes == expectedPayloadSize);

    Modules::DaqiriRxConfigParams params;
    params.interfaceAddress = config.interfaceAddress;
    params.workerCores = config.workerCores;
    params.packetsPerBurst = config.packetsPerBurst;
    params.maxConcurrentBursts = config.maxConcurrentBursts;
    params.packetSizeBytes = impl.validatedPacketSizeBytes;

    daqiri::NetworkConfig networkConfig = {};
    REQUIRE(Modules::BuildDaqiriRxConfig(params, impl.validatedSubscriptions, networkConfig) == Result::SUCCESS);
    REQUIRE(networkConfig.mrs_.at("RX_DATA_0").buf_size_ == expectedPayloadSize);
}

TEST_CASE("ATA receiver rejects overflowing fragment payloads",
          "[stelline][ata_receiver][module][validation]") {
    AtaReceiverValidationImpl impl;
    auto& config = *impl.candidate();
    config = ValidConfig();

    SECTION("Element count overflow") {
        config.partialBlock = {1, std::numeric_limits<U64>::max(), 2, 1};
    }
    SECTION("Byte count overflow") {
        config.partialBlock = {1, std::numeric_limits<U64>::max() / 2 + 1, 1, 1};
    }
    config.totalBlock = config.partialBlock;

    REQUIRE(impl.validate() == Result::ERROR);
}
