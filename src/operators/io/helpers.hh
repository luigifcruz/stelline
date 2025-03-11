#ifndef STELLINE_OPERATORS_IO_HELPERS_HH
#define STELLINE_OPERATORS_IO_HELPERS_HH

#include <stelline/common.hh>

#include "cufile.h"

#ifndef GDS_CHECK_THROW
#define GDS_CHECK_THROW(f, callback) { \
    CUfileError_t status = (f); \
    if (status.err != CU_FILE_SUCCESS) { \
        callback(); \
        HOLOSCAN_LOG_ERROR("[GDS] Error code: {}", status.err); \
        throw std::runtime_error("I/O error."); \
    } \
}
#endif  // GDS_CHECK_THROW

#endif  // STELLINE_OPERATORS_IO_HELPERS_HH
