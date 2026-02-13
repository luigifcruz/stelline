#ifndef STELLINE_CONTEXT_HH
#define STELLINE_CONTEXT_HH

#include <any>
#include <map>
#include <memory>
#include <string>

#include <stelline/common.hh>

namespace stelline {

class STELLINE_API ManifestProvider {
 public:
    ManifestProvider();
    ~ManifestProvider();

    void store(const std::string& key,
               const std::any& value,
               const uint64_t& start = 0,
               const uint64_t& end = UINT64_MAX);
    std::any fetch(const std::string& key, const uint64_t& timestamp = 0) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

struct Metric {
    std::string type;
    std::string value;
};

class STELLINE_API MetricsProvider {
 public:
    using MetricsMap = std::map<std::string, Metric>;

    MetricsProvider();
    ~MetricsProvider();

    void record(const std::string& key,
                const std::string& value,
                const std::string& type = "text",
                const bool& global = false);
    MetricsMap snapshot(const bool& global = false) const;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

//
// Context
//

class STELLINE_API Context {
 public:
    Context();
    ~Context();

    void setManifestProvider(ManifestProvider* provider);
    void setMetricsProvider(MetricsProvider* provider);

    virtual void tick() = 0;
    virtual std::string formatMetrics(const MetricsProvider::MetricsMap& metrics) = 0;

    ManifestProvider* manifest() const;
    MetricsProvider* metrics() const;

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace stelline

#endif  // STELLINE_CONTEXT_HH
