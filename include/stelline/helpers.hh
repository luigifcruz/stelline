#ifndef STELLINE_HELPERS_HH
#define STELLINE_HELPERS_HH

#include <string>
#include <holoscan/holoscan.hpp>

#include <stelline/common.hh>

namespace stelline {

template<typename T>
inline T FetchArg(auto* app, const std::string& handle, const std::string& key) {
    return app->from_config(fmt::format("{}.{}", handle, key)).template as<T>();
}

}  // namespace stelline

#endif  // STELLINE_HELPERS_HH
