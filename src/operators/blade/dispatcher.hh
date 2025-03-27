#ifndef STELLINE_OPERATORS_BLADE_DISPATCHER_HH
#define STELLINE_OPERATORS_BLADE_DISPATCHER_HH

#include <matx.h>

#include <stelline/common.hh>
#include <stelline/utils/juggler.hh>

using namespace Blade;

namespace stelline::operators::blade {

// TODO: Write test for this thing.
class Dispatcher {
 public:
    Dispatcher() = default;

    template<typename T>
    void initialize(const U64& outputPoolSize, const ArrayShape& inputShape) {
        // Resize pools.

        inputPool.resize(2);
        outputPool.resize(2);

        // Reset phases.

        inputPoolPhase = 0;
        outputPoolPhase = 0;

        // Clear in-flight sets.

        inFlightInputs.clear();
        inFlightOutputs.clear();

        // Allocate output tensor pool.

        outputTensorPool.resize(outputPoolSize, [&]{
            auto tensor = matx::make_tensor<T>({
                static_cast<I64>(inputShape.numberOfAspects()),
                static_cast<I64>(inputShape.numberOfFrequencyChannels()),
                static_cast<I64>(inputShape.numberOfTimeSamples()),
                static_cast<I64>(inputShape.numberOfPolarizations())
            }, matx::MATX_DEVICE_MEMORY);
            return std::make_shared<holoscan::Tensor>(tensor.ToDlPack());
        });
    }

    Result run(auto& pipeline,
               auto& receiveCallback,
               auto& convertInputCallback,
               auto& convertOutputCallback,
               auto& emitCallback) {
        // Receive block.

        const auto& block = receiveCallback();

        // Check incoming block timestamp.

        if (timestamp > block.timestamp) {
            HOLOSCAN_LOG_ERROR("Incoming block timestamp is older than the current timestamp.");
            return Result::ERROR;
        }
        timestamp = block.timestamp;

        // Check if in-flight queue state is valid.

        if (inFlightInputs.size() == 2 || inFlightOutputs.size() == 2) {
            HOLOSCAN_LOG_ERROR("Too many in-flight I/O operations ({}/{}).", inFlightInputs.size(), inFlightOutputs.size());
            return Result::ERROR;
        }

        // Enqueue and dequeue block.

        auto inputCallback = [&](){
            // Get input block.
            auto& inputBlock = inputPool[inputPoolPhase];
            inFlightInputs.insert(inputPoolPhase);
            inputPoolPhase = (inputPoolPhase + 1) % 2;

            // Configure job.
            inputBlock.setData(block);

            // Call input callback.
            return convertInputCallback(inputBlock);
        };

        auto resultCallback = [&](){
            return pipeline->transferResult();
        };

        auto outputCallback = [&](){
            // Get output block.
            auto& outputBlock = outputPool[outputPoolPhase];
            inFlightOutputs.insert(outputPoolPhase);
            outputPoolPhase = (outputPoolPhase + 1) % 2;

            // Configure output.
            outputBlock.setMetadata(block);

            // Reserve output buffer.
            if ((outputBlock.tensor = outputTensorPool.get()) == nullptr) {
                HOLOSCAN_LOG_ERROR("Failed to allocate tensor from pool.");
                return Result::ERROR;
            }

            // Call output callback.
            return convertOutputCallback(outputBlock);
        };

        auto dequeueCallback = [&](const U64& inputId, const U64& outputId, const bool& didOutput) {
            // Remove input from in-flight set.
            inFlightInputs.erase(inputId);

            if (didOutput) {
                // Remove output from in-flight set.
                inFlightOutputs.erase(outputId);

                // Send data to the next operator.
                emitCallback(outputPool[outputId]);
            }

            return Result::SUCCESS;
        };

        while (true) {
            const auto& enqueueResult = pipeline->enqueue(inputCallback,
                                                          resultCallback,
                                                          outputCallback,
                                                          inputPoolPhase,
                                                          outputPoolPhase);

            // Block was queued successfully.

            if (enqueueResult == Result::SUCCESS) {
                bool alreadyDequeued = false;

                while (true) {
                    const auto& dequeueResult = pipeline->dequeue(dequeueCallback);

                    // Dequeue succeeded. Check if there are more blocks to dequeue.

                    if (dequeueResult == Result::SUCCESS) {
                        numberOfSuccessfulDequeues++;
                        alreadyDequeued = true;
                        continue;
                    }

                    // Dequeue failed. No more blocks to dequeue yet.

                    if (dequeueResult == Result::RUNNER_QUEUE_NONE_AVAILABLE) {
                        if (!alreadyDequeued) {
                            numberOfPrematureDequeues++;
                        }
                        break;
                    }

                    // Dequeue empty. No more blocks to dequeue.

                    if (dequeueResult == Result::RUNNER_QUEUE_EMPTY) {
                        break;
                    }

                    // Something went wrong with dequeue.

                    return dequeueResult;
                }

                numberOfSuccessfulEnqueues++;
                break;
            }

            // Queue is full. Try to dequeue and try again.

            if (enqueueResult == Result::RUNNER_QUEUE_FULL) {
                while (true) {
                    const auto& dequeueResult = pipeline->dequeue(dequeueCallback);

                    // Dequeue succeeded. Try to enqueue again.

                    if (dequeueResult == Result::SUCCESS) {
                        numberOfSuccessfulDequeues++;
                        break;
                    }

                    // Dequeue failed. Try to dequeue again.

                    if (dequeueResult == Result::RUNNER_QUEUE_NONE_AVAILABLE) {
                        numberOfDequeueRetries++;
                        continue;
                    }

                    // Something went wrong with dequeue.

                    return dequeueResult;
                }

                numberOfFullEnqueues++;
                continue;
            }

            // Something went wrong with enqueue.

            return enqueueResult;
        }

        return Result::SUCCESS;
    };

    void metrics() {
        HOLOSCAN_LOG_INFO("  Queueing Statistics:");
        HOLOSCAN_LOG_INFO("    Successful Enqueues: {}", numberOfSuccessfulEnqueues);
        HOLOSCAN_LOG_INFO("    Successful Dequeues: {}", numberOfSuccessfulDequeues);
        HOLOSCAN_LOG_INFO("    Full Enqueues: {}", numberOfFullEnqueues);
        HOLOSCAN_LOG_INFO("    Dequeue Retries: {}", numberOfDequeueRetries);
        HOLOSCAN_LOG_INFO("    Premature Dequeues: {}", numberOfPrematureDequeues);
    }

 private:
    // State.

    U64 timestamp = 0;

    U64 inputPoolPhase = 0;
    U64 outputPoolPhase = 0;

    std::vector<DspBlock> inputPool;
    std::vector<DspBlock> outputPool;

    std::unordered_set<U64> inFlightInputs;
    std::unordered_set<U64> inFlightOutputs;

    Juggler<holoscan::Tensor> outputTensorPool;

    // Metrics.

    U64 numberOfSuccessfulEnqueues = 0;
    U64 numberOfSuccessfulDequeues = 0;
    U64 numberOfFullEnqueues = 0;
    U64 numberOfDequeueRetries = 0;
    U64 numberOfPrematureDequeues = 0;
};

}  // namespace stelline::operators::blade

#endif  // STELLINE_OPERATORS_BLADE_DISPATCHER_HH
