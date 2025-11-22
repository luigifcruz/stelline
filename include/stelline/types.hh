#ifndef STELLINE_TYPES_HH
#define STELLINE_TYPES_HH

#include <memory>
#include <cstdint>

#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>


namespace stelline {

typedef std::tuple<std::shared_ptr<holoscan::Operator>, std::shared_ptr<holoscan::Operator>> BitInterface;

}  // namespace stelline

#endif  // STELLINE_TYPES_HH
