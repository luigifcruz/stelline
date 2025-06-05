#ifndef STELLINE_BITS_FILESYSTEM_BASE_HH
#define STELLINE_BITS_FILESYSTEM_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/filesystem/base.hh>

namespace stelline::bits::filesystem {

inline BitInterface FilesystemBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::filesystem;

    // Fetch configuration YAML.

    auto mode = FetchNodeArg<std::string>(app, config, "mode");
    auto filePath = FetchNodeArg<std::string>(app, config, "file_path", "./file.bin");

    HOLOSCAN_LOG_INFO("Filesystem Configuration:");
    HOLOSCAN_LOG_INFO("  Mode: {}", mode);
    HOLOSCAN_LOG_INFO("  File Path: {}", filePath);

    // Declare modes.

    auto simple_writer_op = [&](){
        return app->template make_operator<SimpleWriterOp>(
            fmt::format("simple-writer_{}", id),
            Arg("file_path", filePath)
        );
    };

    auto simple_writer_rdma_op = [&](){
        return app->template make_operator<SimpleWriterRdmaOp>(
            fmt::format("simple-writer-rdma_{}", id),
            Arg("file_path", filePath)
        );
    };

    auto dummy_writer_op = [&](){
        return app->template make_operator<DummyWriterOp>(
            fmt::format("dummy-writer_{}", id)
        );
    };

    // Select configuration mode.

    if (mode == "simple_writer") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer operator.");
        const auto& op = simple_writer_op();
        return {op, op};
    }

    if (mode == "simple_writer_rdma") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer RDMA operator.");
        const auto& op = simple_writer_rdma_op();
        return {op, op};
    }

    if (mode == "dummy_writer") {
        HOLOSCAN_LOG_INFO("Creating Dummy Writer operator.");
        const auto& op = dummy_writer_op();
        return {op, op};
    }

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::filesystem

#endif  // STELLINE_BITS_FILESYSTEM_BASE_HH
