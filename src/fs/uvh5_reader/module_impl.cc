#include "module_impl.hh"

namespace Jetstream::Modules {

// UVH5 HDF5 group / dataset paths (pyuvdata / UVH5 specification).
static constexpr const char* kUvh5VisdataDataset = "/Data/visdata";

Result Uvh5ReaderImpl::validate() {
    const auto& config = *candidate();

    if (config.batchSize == 0) {
        JST_ERROR("[MODULE_UVH5_READER] Batch size cannot be zero.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));
    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::create() {
    snapshotTotalBlts.publish(0);
    snapshotCurrentBlt.publish(0);
    snapshotBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (filepath.empty()) {
        JST_WARN("[MODULE_UVH5_READER] File path is empty.");
        return Result::INCOMPLETE;
    }

    // Suppress the default HDF5 error stack printing; we handle errors ourselves.
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);

    // Open the file read-only with the default (POSIX) VFD — no GDS required.
    fileId = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (fileId < 0) {
        JST_WARN("[MODULE_UVH5_READER] Cannot open HDF5 file '{}'.", filepath);
        return Result::INCOMPLETE;
    }

    visDatasetId = H5Dopen2(fileId, kUvh5VisdataDataset, H5P_DEFAULT);
    if (visDatasetId < 0) {
        H5Fclose(fileId);
        fileId = H5I_INVALID_HID;
        JST_WARN("[MODULE_UVH5_READER] Dataset '{}' not found in '{}'.", kUvh5VisdataDataset, filepath);
        return Result::INCOMPLETE;
    }

    visDataspaceId = H5Dget_space(visDatasetId);
    if (visDataspaceId < 0) {
        H5Dclose(visDatasetId);
        H5Fclose(fileId);
        visDatasetId = H5I_INVALID_HID;
        fileId = H5I_INVALID_HID;
        JST_ERROR("[MODULE_UVH5_READER] Failed to get dataspace for '{}'.", kUvh5VisdataDataset);
        return Result::ERROR;
    }

    const int ndims = H5Sget_simple_extent_ndims(visDataspaceId);
    if (ndims != 4) {
        H5Sclose(visDataspaceId);
        H5Dclose(visDatasetId);
        H5Fclose(fileId);
        visDataspaceId = H5I_INVALID_HID;
        visDatasetId = H5I_INVALID_HID;
        fileId = H5I_INVALID_HID;
        JST_ERROR("[MODULE_UVH5_READER] Expected 4-D visdata [Nblts, Nspws, Nfreqs, Npols], got {}.", ndims);
        return Result::ERROR;
    }

    hsize_t dims[4];
    H5Sget_simple_extent_dims(visDataspaceId, dims, nullptr);
    dimNblts = dims[0];
    dimNspws = dims[1];
    dimNfreqs = dims[2];
    dimNpols = dims[3];

    currentBlt = 0;
    snapshotTotalBlts.publish(static_cast<U64>(dimNblts));
    snapshotCurrentBlt.publish(0);

    // Flatten the (usually singleton) Nspws axis into Nfreqs for the output tensor.
    const U64 numChannels = static_cast<U64>(dimNspws) * static_cast<U64>(dimNfreqs);
    const U64 numPols = static_cast<U64>(dimNpols);
    JST_CHECK(buffer.create(device(), DataType::CF32, {batchSize, numChannels, numPols}));
    outputs()["signal"].produced(name(), "signal", buffer);

    JST_INFO("[MODULE_UVH5_READER] Opened '{}': Nblts={} Nspws={} Nfreqs={} Npols={}.",
             filepath, dimNblts, dimNspws, dimNfreqs, dimNpols);

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::destroy() {
    if (visDataspaceId >= 0) { H5Sclose(visDataspaceId); visDataspaceId = H5I_INVALID_HID; }
    if (visDatasetId >= 0)   { H5Dclose(visDatasetId);   visDatasetId = H5I_INVALID_HID; }
    if (fileId >= 0)         { H5Fclose(fileId);         fileId = H5I_INVALID_HID; }

    snapshotBandwidth.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

Result Uvh5ReaderImpl::reconfigure() {
    const auto& config = *candidate();

    if (config.filepath != filepath ||
        config.batchSize != batchSize) {
        return Result::RECREATE;
    }

    loop = config.loop;
    playing = config.playing;
    return Result::SUCCESS;
}

U64 Uvh5ReaderImpl::getCurrentBlt() const {
    return snapshotCurrentBlt.get();
}

U64 Uvh5ReaderImpl::getTotalBlts() const {
    return snapshotTotalBlts.get();
}

F32 Uvh5ReaderImpl::getCurrentBandwidth() const {
    return snapshotBandwidth.get();
}

void Uvh5ReaderImpl::updateBandwidth(const U64 deltaBytes) {
    constexpr double kPeriodSeconds = 0.10;
    constexpr double kEmaAlpha = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsed < kPeriodSeconds) {
        return;
    }

    const double instant = static_cast<double>(bytesSinceLastMeasurement) /
                           (1024.0 * 1024.0) / elapsed;
    const double smoothed = kEmaAlpha * instant +
                            (1.0 - kEmaAlpha) * snapshotBandwidth.get();
    snapshotBandwidth.publish(static_cast<F32>(smoothed));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
