#include "module_impl.hh"

namespace Jetstream::Modules {

Result DummyWriterImpl::validate() {
    return Result::SUCCESS;
}

Result DummyWriterImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));

    return Result::SUCCESS;
}

Result DummyWriterImpl::create() {
    startTime = std::chrono::steady_clock::now();
    iterationCount = 0;
    latestTimestamp = 0;

    return Result::SUCCESS;
}

Result DummyWriterImpl::destroy() {
    return Result::SUCCESS;
}

const U64& DummyWriterImpl::getIterationCount() const {
    return iterationCount;
}

F64 DummyWriterImpl::getAverageDurationMs() const {
    if (iterationCount == 0) {
        return 0.0;
    }

    const auto elapsed = std::chrono::steady_clock::now() - startTime;
    const auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    return static_cast<F64>(elapsedMs) / static_cast<F64>(iterationCount);
}

const U64& DummyWriterImpl::getLatestTimestamp() const {
    return latestTimestamp;
}

}  // namespace Jetstream::Modules
