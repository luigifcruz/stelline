#pragma once

#include <hdf5.h>

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
