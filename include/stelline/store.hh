#ifndef STELLINE_STORE_HH
#define STELLINE_STORE_HH

#include <map>
#include <string>

#include <stelline/common.hh>

namespace stelline {

class STELLINE_API StoreInterface {
 public:
    using MetricsMap = std::map<std::string, std::string>;

    virtual ~StoreInterface() = default;

    virtual MetricsMap collectMetricsMap() = 0;
    virtual std::string collectMetricsString() = 0;
};

}  // namespace stelline

#endif  // STELLINE_STORE_HH
