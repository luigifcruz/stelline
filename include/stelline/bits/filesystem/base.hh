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
    auto file_path = FetchNodeArg<std::string>(app, config, "file_path", "./file.bin");

    HOLOSCAN_LOG_INFO("Filesystem Configuration:");
    HOLOSCAN_LOG_INFO("  Mode: {}", mode);
    HOLOSCAN_LOG_INFO("  File Path: {}", file_path);

    // Declare modes.

    auto simple_writer_op_cb = [&](const auto& op_id){
        return app->template make_operator<SimpleWriterOp>(
            op_id,
            Arg("file_path", file_path)
        );
    };

    auto simple_writer_rdma_op_cb = [&](const auto& op_id){
        return app->template make_operator<SimpleWriterRdmaOp>(
            op_id,
            Arg("file_path", file_path)
        );
    };

    auto dummy_writer_op_cb = [&](const auto& op_id){
        return app->template make_operator<DummyWriterOp>(
            op_id
        );
    };

    // Select configuration mode.

    if (mode == "simple_writer") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer operator.");
        const auto& simple_writer_id = fmt::format("filesystem-simple-writer-{}", id);
        auto simple_writer_op = simple_writer_op_cb(simple_writer_id);

        return {simple_writer_op, simple_writer_op};
    }

    if (mode == "simple_writer_rdma") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer RDMA operator.");
        const auto& simple_writer_rdma_id = fmt::format("filesystem-simple-writer-rdma-{}", id);
        auto simple_writer_rdma_op = simple_writer_rdma_op_cb(simple_writer_rdma_id);

        return {simple_writer_rdma_op, simple_writer_rdma_op};
    }

    if (mode == "dummy_writer") {
        HOLOSCAN_LOG_INFO("Creating Dummy Writer operator.");
        const auto& dummy_writer_id = fmt::format("filesystem-dummy-writer-{}", id);
        auto dummy_writer_op = dummy_writer_op_cb(dummy_writer_id);

        return {dummy_writer_op, dummy_writer_op};
    }


#ifdef STELLINE_LOADER_FBH5
    auto fbh5_writer_rdma_op_cb = [&](const auto& op_id){
        return app->template make_operator<Fbh5WriterRdmaOp>(
            op_id,
            Arg("file_path", file_path)
        );
    };

    if (mode == "fbh5_writer_rdma") {
        HOLOSCAN_LOG_INFO("Creating FBH5 Writer RDMA operator.");
        const auto& fbh5_writer_rdma_id = fmt::format("filesystem-fbh5-writer-rdma-{}", id);
        auto fbh5_writer_rdma_op = fbh5_writer_rdma_op_cb(fbh5_writer_rdma_id);

        return {fbh5_writer_rdma_op, fbh5_writer_rdma_op};
    }
#endif
#ifdef STELLINE_LOADER_UVH5
    auto telinfo_file_path = FetchNodeArg<std::string>(app, config, "telinfo_file_path");
    auto obsantinfo_file_path = FetchNodeArg<std::string>(app, config, "obsantinfo_file_path");
    auto iers_file_path = FetchNodeArg<std::string>(app, config, "iers_file_path");
    HOLOSCAN_LOG_INFO("  Telinfo Path: {}", telinfo_file_path);
    HOLOSCAN_LOG_INFO("  Obsantinfo Path: {}", obsantinfo_file_path);
    HOLOSCAN_LOG_INFO("  IERS Path: {}", iers_file_path);
    auto uvh5_writer_rdma_op_cb = [&](const auto& op_id){
        return app->template make_operator<Uvh5WriterRdmaOp>(
            op_id,
            Arg("output_file_path", file_path),
            Arg("telinfo_file_path", telinfo_file_path),
            Arg("obsantinfo_file_path", obsantinfo_file_path),
            Arg("iers_file_path", iers_file_path)
        );
    };

    if (mode == "uvh5_writer_rdma") {
        HOLOSCAN_LOG_INFO("Creating FBH5 Writer RDMA operator.");
        const auto& uvh5_writer_rdma_id = fmt::format("filesystem-fbh5-writer-rdma-{}", id);
        auto uvh5_writer_rdma_op = uvh5_writer_rdma_op_cb(uvh5_writer_rdma_id);

        return {uvh5_writer_rdma_op, uvh5_writer_rdma_op};
    }
#endif

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::filesystem

#endif  // STELLINE_BITS_FILESYSTEM_BASE_HH
