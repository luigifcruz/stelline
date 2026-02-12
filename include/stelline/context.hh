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

    std::any pull(const std::string& key, uint64_t timestamp);

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

class STELLINE_API MetricsProvider {
 public:
    using MetricsMap = std::map<std::string, std::string>;

    MetricsProvider();
    ~MetricsProvider();

    void push(const std::string& key, const std::string& value, bool local = false);

    MetricsMap collect();

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

 protected:
    ManifestProvider* manifest();
    MetricsProvider* metrics();

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace stelline

#endif  // STELLINE_CONTEXT_HH
