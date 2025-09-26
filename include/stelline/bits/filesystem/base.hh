#ifndef STELLINE_BITS_FILESYSTEM_BASE_HH
#define STELLINE_BITS_FILESYSTEM_BASE_HH

#include <holoscan/holoscan.hpp>

#include <stelline/helpers.hh>
#include <stelline/operators/filesystem/base.hh>

namespace stelline::bits::filesystem {

inline BitInterface FilesystemBit(auto* app, auto& pool, uint64_t id, const std::string& config) {
    using namespace holoscan;
    using namespace stelline::operators::filesystem;

    // Create metadata storage.

    auto metadata = std::make_shared<MetadataStorage>();

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

    auto hdf5_writer_rdma_op_cb = [&](const auto& op_id){
        return app->template make_operator<Hdf5WriterRdmaOp>(
            op_id,
            Arg("file_path", file_path)
        );
    };

    // Select configuration mode.

    if (mode == "simple_writer") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer operator.");
        const auto& simple_writer_id = fmt::format("filesystem-simple-writer-{}", id);
        auto simple_writer_op = simple_writer_op_cb(simple_writer_id);
        simple_writer_op->load_metadata(simple_writer_id, metadata);
        return {simple_writer_op, simple_writer_op, simple_writer_op};
    }

    if (mode == "simple_writer_rdma") {
        HOLOSCAN_LOG_INFO("Creating Simple Writer RDMA operator.");
        const auto& simple_writer_rdma_id = fmt::format("filesystem-simple-writer-rdma-{}", id);
        auto simple_writer_rdma_op = simple_writer_rdma_op_cb(simple_writer_rdma_id);
        simple_writer_rdma_op->load_metadata(simple_writer_rdma_id, metadata);
        return {simple_writer_rdma_op, simple_writer_rdma_op, simple_writer_rdma_op};
    }

    if (mode == "dummy_writer") {
        HOLOSCAN_LOG_INFO("Creating Dummy Writer operator.");
        const auto& dummy_writer_id = fmt::format("filesystem-dummy-writer-{}", id);
        auto dummy_writer_op = dummy_writer_op_cb(dummy_writer_id);
        dummy_writer_op->load_metadata(dummy_writer_id, metadata);
        return {dummy_writer_op, dummy_writer_op, dummy_writer_op};
    }

#ifdef STELLINE_LOADER_HDF5
    if (mode == "hdf5_writer_rdma") {
        HOLOSCAN_LOG_INFO("Creating HDF5 Writer RDMA operator.");
        const auto& hdf5_writer_rdma_id = fmt::format("filesystem-hdf5-writer-rdma-{}", id);
        auto hdf5_writer_rdma_op = hdf5_writer_rdma_op_cb(hdf5_writer_rdma_id);
        hdf5_writer_rdma_op->load_metadata(hdf5_writer_rdma_id, metadata);
        return {hdf5_writer_rdma_op, hdf5_writer_rdma_op, hdf5_writer_rdma_op};
    }
#endif

    HOLOSCAN_LOG_ERROR("Unsupported mode: {}", mode);
    throw std::runtime_error("Unsupported mode.");
}

}  // namespace stelline::bits::filesystem

#endif  // STELLINE_BITS_FILESYSTEM_BASE_HH
