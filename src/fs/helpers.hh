#pragma once

#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <cerrno>
#include <filesystem>
#include <random>
#include <system_error>

#include <hdf5.h>

#include <jetstream/logger.hh>
#include <jetstream/types.hh>

#ifndef JST_HDF5_CHECK
#define JST_HDF5_CHECK(x, callback) { \
    herr_t val = static_cast<herr_t>((x)); \
    if (val < 0) { \
        const herr_t err = val; \
        callback(); \
        return Result::ERROR; \
    } \
}
#endif  // JST_HDF5_CHECK

namespace Jetstream::Modules {

struct RecordingGrid {
    U64 startSample = 0;
    U64 synctime = 0;
    U64 samplesPerRow = 0;
};

inline herr_t WriteRecordingGrid(const hid_t location, const bool datasets, const RecordingGrid& grid) {
    const auto write = [&](const char* name, const hid_t type, const void* value) -> herr_t {
        const hid_t space = H5Screate(H5S_SCALAR);
        if (space < 0) {
            return -1;
        }
        const hid_t id = datasets
                             ? H5Dcreate2(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
                             : H5Acreate2(location, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
        herr_t status = -1;
        if (id >= 0) {
            status = datasets ? H5Dwrite(id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value) : H5Awrite(id, type, value);
            if ((datasets ? H5Dclose(id) : H5Aclose(id)) < 0) {
                status = -1;
            }
        }
        if (H5Sclose(space) < 0) {
            status = -1;
        }
        return status;
    };

    if (write("rf_start_sample", H5T_NATIVE_UINT64, &grid.startSample) < 0 ||
        write("synctime", H5T_NATIVE_UINT64, &grid.synctime) < 0 ||
        write("rf_samples_per_row", H5T_NATIVE_UINT64, &grid.samplesPerRow) < 0) {
        return -1;
    }
    return 0;
}

inline void RemovePartialRecording(std::filesystem::path& partial) {
    if (!partial.empty()) {
        std::error_code ec;
        std::filesystem::remove(partial, ec);
        partial.clear();
    }
}

inline std::error_code ReservePartialRecording(const std::filesystem::path& path,
                                               std::filesystem::path& partial) {
    std::random_device random;
    for (int attempt = 0; attempt < 16; ++attempt) {
        const U64 token = (static_cast<U64>(random()) << 32) ^ random();
        auto candidate = path;
        candidate += jst::fmt::format(".{:016x}.partial", token);
        const int fd = ::open(candidate.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, 0666);
        if (fd >= 0) {
            ::close(fd);
            partial = std::move(candidate);
            return {};
        }
        if (errno != EEXIST) {
            return {errno, std::generic_category()};
        }
    }
    return std::make_error_code(std::errc::file_exists);
}

inline std::error_code PublishPartialRecording(std::filesystem::path& partial,
                                               const std::filesystem::path& path,
                                               const bool overwrite) {
    if (partial.empty()) {
        return {};
    }
    if (::renameat2(AT_FDCWD, partial.c_str(), AT_FDCWD, path.c_str(), overwrite ? 0 : RENAME_NOREPLACE) != 0) {
        if (overwrite || errno != EINVAL) {
            return {errno, std::generic_category()};
        }
        if (::link(partial.c_str(), path.c_str()) != 0) {
            return {errno, std::generic_category()};
        }
        RemovePartialRecording(partial);
        return {};
    }
    partial.clear();
    return {};
}

}  // namespace Jetstream::Modules
