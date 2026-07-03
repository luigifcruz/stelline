#ifndef STELLINE_NEXUS_HH
#define STELLINE_NEXUS_HH

#include <any>
#include <limits>
#include <memory>
#include <string>

#include <jetstream/block_interface.hh>
#include <jetstream/types.hh>

namespace stelline {

struct NexusBridge;

class Nexus {
 public:
    using MetricsList = Jetstream::Block::Interface::EntryList;

    static bool Available();

    static bool MetadataContains(const std::string& key,
                                 Jetstream::U64 timestamp = std::numeric_limits<Jetstream::U64>::min());
    static std::any MetadataFetch(const std::string& key,
                                  Jetstream::U64 timestamp = std::numeric_limits<Jetstream::U64>::min());

    template<typename T>
    static bool TryMetadataFetch(const std::string& key,
                                 T& value,
                                 Jetstream::U64 timestamp = std::numeric_limits<Jetstream::U64>::min()) {
        if (!MetadataContains(key, timestamp)) {
            return false;
        }

        try {
            value = std::any_cast<T>(MetadataFetch(key, timestamp));
            return true;
        } catch (const std::bad_any_cast&) {
            return false;
        }
    }

    Nexus(const Nexus&) = delete;
    Nexus& operator=(const Nexus&) = delete;
    Nexus(Nexus&&) = delete;
    Nexus& operator=(Nexus&&) = delete;

 protected:
    static void Initialize();
    static void Deinitialize();

    static void MetadataStore(const std::string& key,
                              const std::any& value,
                              Jetstream::U64 start = std::numeric_limits<Jetstream::U64>::min(),
                              Jetstream::U64 end = std::numeric_limits<Jetstream::U64>::max());
    static void MetadataClear();
    static Jetstream::U64 MetadataSize();

    static void SignalStatus(const std::string& status, const std::string& log = {});

    static void MetricsPush(const std::string& blockName, const MetricsList& metrics);
    static void MetricsFlush();

    friend struct NexusBridge;

 private:
    static Nexus& Instance();

    Nexus();
    ~Nexus();

    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

}  // namespace stelline

#endif  // STELLINE_NEXUS_HH
