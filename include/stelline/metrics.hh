#ifndef STELLINE_METRICS_HH
#define STELLINE_METRICS_HH

#include <any>
#include <map>
#include <string>

#include <stelline/common.hh>

namespace stelline {

class STELLINE_API MetricsInterface {
 public:
    using MetricsMap = std::map<std::string, std::string>;

    virtual ~MetricsInterface() = default;

    virtual MetricsMap collectMetricsMap() = 0;
    virtual std::string collectMetricsString() = 0;
};

}  // namespace stelline

#endif  // STELLINE_METRICS_HH
