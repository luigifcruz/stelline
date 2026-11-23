#pragma once

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <jetstream/backend/base.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/flowgraph.hh>
#include <jetstream/flowgraph_environment.hh>
#include <jetstream/module.hh>
#include <jetstream/parser.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/types.hh>

namespace Jetstream::Testing {

struct BackendGuard {
    ~BackendGuard() {
        Backend::DestroyAll();
    }
};

struct ScratchDirectory {
    std::filesystem::path path;

    ScratchDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "stelline-test-XXXXXX").string();
        REQUIRE(mkdtemp(pattern.data()) != nullptr);
        path = pattern;
    }

    ~ScratchDirectory() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct TextValue {
    std::string value;
    JST_SERDES(value);
};

struct RealValue {
    F64 value = 0.0;
    JST_SERDES(value);
};

struct IntegerValue {
    I64 value = 0;
    JST_SERDES(value);
};

inline void SetEnvironment(Flowgraph& flowgraph, const std::string& key, const std::string& value) {
    REQUIRE(flowgraph.environment().set(key, TextValue{value}) == Result::SUCCESS);
}

inline void SetEnvironment(Flowgraph& flowgraph, const std::string& key, const F64 value) {
    REQUIRE(flowgraph.environment().set(key, RealValue{value}) == Result::SUCCESS);
}

inline void SetEnvironment(Flowgraph& flowgraph, const std::string& key, const I64 value) {
    REQUIRE(flowgraph.environment().set(key, IntegerValue{value}) == Result::SUCCESS);
}

inline int RunProgram(const std::vector<std::string>& arguments) {
    std::vector<char*> argv;
    for (const auto& argument : arguments) {
        argv.push_back(const_cast<char*>(argument.c_str()));
    }
    argv.push_back(nullptr);

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addopen(&actions, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&actions, 2, "/dev/null", O_WRONLY, 0);
    pid_t pid = 0;
    const int spawned = posix_spawn(&pid, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    if (spawned != 0) {
        return -1;
    }
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) {
        return -1;
    }
    return WEXITSTATUS(status);
}

struct CpuModule {
    std::shared_ptr<Module> module;
    NativeCpuRuntimeContext* runtime = nullptr;

    template<typename Config>
    CpuModule(const std::string& type, const Config& config, const TensorMap& inputs) {
        REQUIRE(Registry::BuildModule(type, DeviceType::CPU, RuntimeType::NATIVE, "generic", module) ==
                Result::SUCCESS);
        REQUIRE(module->create(type, config, inputs) == Result::SUCCESS);
        runtime = module->getImpl<NativeCpuRuntimeContext>();
        REQUIRE(runtime != nullptr);
        REQUIRE(runtime->computeInitialize() == Result::SUCCESS);
    }

    ~CpuModule() {
        if (runtime) {
            (void)runtime->computeDeinitialize();
        }
        if (module) {
            (void)module->destroy();
        }
    }

    Result compute() {
        return runtime->computeSubmit();
    }

    Tensor output(const std::string& name = "buffer") const {
        return module->outputs().at(name).tensor;
    }
};

}  // namespace Jetstream::Testing
