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

    struct Job {
        DspBlock input;
        DspBlock output;
        U64 iterations;
    };

    template<typename T>
    void initialize(const U64& outputPoolSize, const ArrayShape& inputShape) {
        // Resize jobs.

        jobs.resize(2);

        // Allocate output tensor pool.

        outputTensorPool.resize(outputPoolSize, [&]{
            auto tensor = matx::make_tensor<T>({
                static_cast<I64>(inputShape.numberOfAspects()),
                static_cast<I64>(inputShape.numberOfFrequencyChannels()),
                static_cast<I64>(inputShape.numberOfTimeSamples()),
                static_cast<I64>(inputShape.numberOfPolarizations())
            });
            return std::make_shared<holoscan::Tensor>(tensor.GetDLPackTensor());
        });
    }

    void run(auto& pipeline, 
             auto& receiveCallback,
             auto& convertInputCallback, 
             auto& convertOutputCallback,
             auto& emitCallback) {
        {
            auto& job = jobs[iterations % 2];

            auto inputCallback = [&](){
                const auto& block = receiveCallback();

                // Configure job.

                if (job.iterations == 0) {
                    job.output.setMetadata(block);
                    while ((job.output.tensor = outputTensorPool.get()) == nullptr) {
                        throw std::runtime_error("Failed to allocate tensor from pool.");
                    }
                }
                job.input.setData(block);
                job.iterations++;

                // Call input callback.
                return convertInputCallback(job);
            };
            auto resultCallback = [&](){
                return pipeline->transferResult();
            };
            auto outputCallback = [&](){
                // Incremement iteration count.

                iterations++;

                // Call output callback.
                return convertOutputCallback(job);
            };
            pipeline->enqueue(inputCallback, resultCallback, outputCallback, 0, iterations);
        }

        pipeline->dequeue([&](const U64& inputId, 
                              const U64& outputId,
                              const bool& didOutput){
            if (didOutput) {
                // Get job reference.
                auto& job = jobs[outputId % 2];

                // Send data to the next operator.
                emitCallback(job);

                // Reset job.

                job.iterations = 0;
                job.input = {};
                job.output = {};
            }
            return Result::SUCCESS;
        });
    };

 private:
    U64 iterations;
    std::vector<Job> jobs;

    Juggler<holoscan::Tensor> outputTensorPool;
};

}  // namespace stelline::operators::blade

#endif  // STELLINE_OPERATORS_BLADE_DISPATCHER_HH
