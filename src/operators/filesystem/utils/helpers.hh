#ifndef STELLINE_OPERATORS_FILESYSTEM_HELPERS_HH
#define STELLINE_OPERATORS_FILESYSTEM_HELPERS_HH

#include <stelline/common.hh>

//
// CUDA
//

#include <cuda_runtime.h>

#ifndef CUDA_CHECK_THROW
#define CUDA_CHECK_THROW(f, callback) { \
    cudaError_t status = (f); \
    if (status != cudaSuccess) { \
        callback(); \
        HOLOSCAN_LOG_ERROR("[CUDA] Error code: {} ({})", static_cast<int>(status), cudaGetErrorString(status)); \
        throw std::runtime_error("CUDA error."); \
    } \
}
#endif  // CUDA_CHECK_THROW

//
// GDS
//

#include <cufile.h>

#ifndef GDS_CHECK_THROW
#define GDS_CHECK_THROW(f, callback) { \
    CUfileError_t status = (f); \
    if (status.err != CU_FILE_SUCCESS) { \
        callback(); \
        HOLOSCAN_LOG_ERROR("[GDS] Error code: {}", static_cast<int>(status.err)); \
        throw std::runtime_error("GDS I/O error."); \
    } \
}
#endif  // GDS_CHECK_THROW

#ifdef STELLINE_LOADER_HDF5
//
// HDF
//

#include <hdf5.h>
#include <H5FDgds.h>

#ifndef HDF5_CHECK_THROW
#define HDF5_CHECK_THROW(f, callback) { \
    herr_t status = (f); \
    if (status < 0) { \
        callback(); \
        HOLOSCAN_LOG_ERROR("[HDF5] Error code: {}", status); \
        throw std::runtime_error("HDF5 I/O error."); \
    } \
}
#endif  // HDF5_CHECK_THROW
#endif  // STELLINE_LOADER_HDF5

#endif  // STELLINE_OPERATORS_FILESYSTEM_HELPERS_HH
