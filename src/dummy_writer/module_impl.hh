#ifndef STELLINE_DUMMY_WRITER_MODULE_IMPL_HH
#define STELLINE_DUMMY_WRITER_MODULE_IMPL_HH

#include <chrono>

#include <stelline/dummy_writer/module.hh>
#include <jetstream/detail/module_impl.hh>

namespace Jetstream::Modules {

struct DummyWriterImpl : public Module::Impl, public DynamicConfig<DummyWriter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    const U64& getIterationCount() const;
    F64 getAverageDurationMs() const;
    const U64& getLatestTimestamp() const;

 protected:
    U64 iterationCount = 0;
    U64 latestTimestamp = 0;
    std::chrono::steady_clock::time_point startTime;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DUMMY_WRITER_MODULE_IMPL_HH
