#ifndef STELLINE_BITS_IO_BASE_HH
#define STELLINE_BITS_IO_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/io/base.hh>

namespace stelline::bits::frbnn {

inline BitInterface IoSinkBit(auto* app, auto& pool, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::io;

    // Fetch configuration YAML.

    auto mode = FetchNodeArg<std::string>(app, config, "mode");
    auto enableRdma = FetchNodeArg<bool>(app, config, "enable_rdma");
    auto filePath = FetchNodeArg<std::string>(app, config, "file_path");

    HOLOSCAN_LOG_INFO("I/O Sink Configuration:");
    HOLOSCAN_LOG_INFO("  Mode: {}", mode);
    HOLOSCAN_LOG_INFO("  Enable RDMA: {}", enableRdma);
    HOLOSCAN_LOG_INFO("  File Path: {}", filePath);

    // Declare modes.

    auto simple_writer_op = [&](){
        return app->template make_operator<SimpleSinkOp>(
            "simple-writer",
            Arg("enable_rdma", enableRdma),
            Arg("file_path", filePath)
        );
    };

    // Select configuration mode.

    if (mode == "simple_writer") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer operator.");
        const auto& op = simple_writer_op();
        return {op, op};
    }

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::io

#endif  // STELLINE_BITS_IO_BASE_HH
