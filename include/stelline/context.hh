#ifndef STELLINE_CONTEXT_HH
#define STELLINE_CONTEXT_HH

#include <any>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

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

    template <typename T>
    bool fetch(const std::string& key, T& out, const uint64_t& timestamp = 0) const {
        const std::any val = fetchAny(key, timestamp);
        if (!val.has_value()) {
            throw std::runtime_error("Manifest key not found: " + key);
        }
        using U = std::decay_t<T>;
        if constexpr (std::is_same_v<U, std::any>) {
            out = val;
            return true;
        }
        if (auto* ptr = std::any_cast<U>(&val)) {
            out = *ptr;
            return true;
        }
        throw std::runtime_error("Manifest key has unexpected type: " + key);
    }

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;

    std::any fetchAny(const std::string& key, const uint64_t& timestamp = 0) const;
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
