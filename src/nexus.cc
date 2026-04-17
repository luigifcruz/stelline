#include <stelline/nexus.hh>

#include <jetstream/logger.hh>
#include <jetstream/parser.hh>

#include <algorithm>
#include <any>
#include <deque>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {

struct StellineManifestView {
    const char* key;
    const char* valueType;
    const char* value;
    const char* start;
    const char* end;
};

struct StellineMetricView {
    const char* block;
    const char* key;
    const char* label;
    const char* format;
    const char* value;
};

struct StellineStatusView {
    const char* status;
    const char* log;
};

}

namespace stelline {

namespace {

static bool IsMetricFormat(const std::string& format) {
    return format == "stelline-metrics" ||
           format == "stelline-metrics-global-number" ||
           format == "stelline-metrics-global-string";
}

template<typename T>
static bool DecodeWithParser(const std::string& encoded, std::any& value) {
    T typed{};
    if (Jetstream::Parser::StringToTyped<T>(encoded, typed) != Jetstream::Result::SUCCESS) {
        return false;
    }

    value = typed;
    return true;
}

}  // namespace

struct Nexus::Impl {
    struct ManifestEntry {
        Jetstream::U64 start = std::numeric_limits<Jetstream::U64>::min();
        Jetstream::U64 end = std::numeric_limits<Jetstream::U64>::max();
        std::any value;
    };

    struct MetricEntry {
        std::string block;
        std::string key;
        std::string label;
        std::string format;
        std::string value;
    };

    struct StatusEntry {
        std::string status;
        std::string log;
    };

    bool available() const;
    bool metadataContains(const std::string& key, Jetstream::U64 timestamp) const;
    std::any metadataFetch(const std::string& key, Jetstream::U64 timestamp) const;
    void metadataStore(const std::string& key,
                       const std::any& value,
                       Jetstream::U64 start,
                       Jetstream::U64 end);
    void metadataClear();
    Jetstream::U64 metadataSize() const;

    void initialize();
    void deinitialize();
    void signalStatus(const std::string& status, const std::string& log);
    void metricsPush(const std::string& blockName, const Nexus::MetricsList& metrics);
    void metricsFlush();

    void reset();
    void beginManifestUpdate();
    int pushManifestEntry(const StellineManifestView& view);
    void commitManifestUpdate(bool connected);

    bool popMetric();
    bool popStatus();

    int readMetric(StellineMetricView& view);
    int readStatus(StellineStatusView& view);

    bool decodeTypedValue(const std::string& type, const std::string& encoded, std::any& value) const;

    mutable std::mutex stateMutex;
    bool bridgeConnected = false;
    bool initialManifestReceived = false;

    mutable std::mutex manifestMutex;
    std::unordered_map<std::string, std::vector<ManifestEntry>> manifestCache;
    std::unordered_map<std::string, std::vector<ManifestEntry>> stagingManifestCache;

    mutable std::mutex metricsMutex;
    std::unordered_map<std::string, std::vector<MetricEntry>> pendingMetrics;
    std::deque<MetricEntry> metricQueue;
    MetricEntry currentMetric;

    mutable std::mutex statusMutex;
    std::deque<StatusEntry> statusQueue;
    StatusEntry currentStatus;
};

struct NexusBridge {
    static Nexus::Impl& impl() {
        return *Nexus::Instance().pimpl;
    }
};

Nexus::Nexus() {
    pimpl = std::make_unique<Impl>();
}

Nexus::~Nexus() {
    pimpl.reset();
}

bool Nexus::Impl::available() const {
    std::lock_guard<std::mutex> guard(stateMutex);
    return bridgeConnected && initialManifestReceived;
}

bool Nexus::Impl::metadataContains(const std::string& key, const Jetstream::U64 timestamp) const {
    std::lock_guard<std::mutex> guard(manifestMutex);
    const auto entryIt = manifestCache.find(key);
    if (entryIt == manifestCache.end()) {
        return false;
    }

    return std::any_of(entryIt->second.begin(), entryIt->second.end(), [timestamp](const ManifestEntry& entry) {
        return timestamp >= entry.start && timestamp < entry.end;
    });
}

std::any Nexus::Impl::metadataFetch(const std::string& key, const Jetstream::U64 timestamp) const {
    std::lock_guard<std::mutex> guard(manifestMutex);
    const auto entryIt = manifestCache.find(key);
    if (entryIt == manifestCache.end()) {
        return {};
    }

    const auto match = std::find_if(entryIt->second.begin(), entryIt->second.end(), [timestamp](const ManifestEntry& entry) {
        return timestamp >= entry.start && timestamp < entry.end;
    });

    return match != entryIt->second.end() ? match->value : std::any{};
}

void Nexus::Impl::metadataStore(const std::string& key,
                                const std::any& value,
                                const Jetstream::U64 start,
                                const Jetstream::U64 end) {
    ManifestEntry nextEntry;
    nextEntry.start = start;
    nextEntry.end = end;
    nextEntry.value = value;

    std::lock_guard<std::mutex> guard(manifestMutex);
    auto& entries = manifestCache[key];
    const auto existing = std::find_if(entries.begin(), entries.end(), [start, end](const ManifestEntry& entry) {
        return entry.start == start && entry.end == end;
    });

    if (existing != entries.end()) {
        *existing = nextEntry;
        return;
    }

    entries.push_back(std::move(nextEntry));
}

void Nexus::Impl::metadataClear() {
    std::lock_guard<std::mutex> guard(manifestMutex);
    manifestCache.clear();
}

Jetstream::U64 Nexus::Impl::metadataSize() const {
    std::lock_guard<std::mutex> guard(manifestMutex);

    Jetstream::U64 count = 0;
    for (const auto& [_, entries] : manifestCache) {
        count += static_cast<Jetstream::U64>(entries.size());
    }

    return count;
}

void Nexus::Impl::initialize() {
}

void Nexus::Impl::deinitialize() {
    {
        std::lock_guard<std::mutex> guard(stateMutex);
        bridgeConnected = false;
        initialManifestReceived = false;
    }

    {
        std::lock_guard<std::mutex> guard(manifestMutex);
        manifestCache.clear();
        stagingManifestCache.clear();
    }

    {
        std::lock_guard<std::mutex> guard(metricsMutex);
        pendingMetrics.clear();
    }
}

void Nexus::Impl::signalStatus(const std::string& status, const std::string& log) {
    std::lock_guard<std::mutex> guard(statusMutex);
    statusQueue.push_back({
        .status = status,
        .log = log,
    });
}

void Nexus::Impl::metricsPush(const std::string& blockName, const Nexus::MetricsList& metrics) {
    std::vector<MetricEntry> encodedMetrics;
    encodedMetrics.reserve(metrics.size());

    for (const auto& [metricKey, entry] : metrics) {
        if (!IsMetricFormat(entry.format) || !entry.metric) {
            continue;
        }

        std::string value;
        try {
            value = std::any_cast<std::string>(entry.metric());
        } catch (const std::bad_any_cast&) {
            JST_WARN("[NEXUS] Metric '{}.{}' did not produce a string value.", blockName, metricKey);
            continue;
        } catch (const std::exception& error) {
            JST_WARN("[NEXUS] Failed to evaluate metric '{}.{}': {}.", blockName, metricKey, error.what());
            continue;
        } catch (...) {
            JST_WARN("[NEXUS] Failed to evaluate metric '{}.{}' due to an unknown error.",
                     blockName,
                     metricKey);
            continue;
        }

        encodedMetrics.push_back({
            .block = blockName,
            .key = metricKey,
            .label = entry.label,
            .format = entry.format,
            .value = std::move(value),
        });
    }

    std::lock_guard<std::mutex> guard(metricsMutex);
    if (encodedMetrics.empty()) {
        pendingMetrics.erase(blockName);
        return;
    }

    pendingMetrics[blockName] = std::move(encodedMetrics);
}

void Nexus::Impl::metricsFlush() {
    std::unordered_map<std::string, std::vector<MetricEntry>> metricsBatch;

    {
        std::lock_guard<std::mutex> guard(metricsMutex);
        if (pendingMetrics.empty()) {
            return;
        }

        metricsBatch = std::move(pendingMetrics);
        pendingMetrics.clear();

        for (const auto& [_, metrics] : metricsBatch) {
            for (const auto& metric : metrics) {
                metricQueue.push_back(metric);
            }
        }
    }
}

void Nexus::Impl::reset() {
    {
        std::lock_guard<std::mutex> guard(stateMutex);
        bridgeConnected = false;
        initialManifestReceived = false;
    }

    {
        std::lock_guard<std::mutex> guard(manifestMutex);
        manifestCache.clear();
        stagingManifestCache.clear();
    }

    {
        std::lock_guard<std::mutex> guard(metricsMutex);
        pendingMetrics.clear();
        metricQueue.clear();
        currentMetric = {};
    }

    {
        std::lock_guard<std::mutex> guard(statusMutex);
        statusQueue.clear();
        currentStatus = {};
    }
}

void Nexus::Impl::beginManifestUpdate() {
    std::lock_guard<std::mutex> guard(manifestMutex);
    stagingManifestCache.clear();
}

int Nexus::Impl::pushManifestEntry(const StellineManifestView& view) {
    if (!view.key || !view.valueType || !view.value || !view.start || !view.end) {
        JST_WARN("[NEXUS] Received an incomplete manifest entry over FFI.");
        return 0;
    }

    const std::string keyString = view.key;
    if (keyString.empty()) {
        JST_WARN("[NEXUS] Received a manifest entry without a key over FFI.");
        return 0;
    }

    ManifestEntry nextEntry;
    if (Jetstream::Parser::StringToTyped<Jetstream::U64>(view.start, nextEntry.start) != Jetstream::Result::SUCCESS) {
        JST_WARN("[NEXUS] Failed to parse the manifest start for '{}'.", keyString);
        return 0;
    }

    if (Jetstream::Parser::StringToTyped<Jetstream::U64>(view.end, nextEntry.end) != Jetstream::Result::SUCCESS) {
        JST_WARN("[NEXUS] Failed to parse the manifest end for '{}'.", keyString);
        return 0;
    }

    if (!decodeTypedValue(view.valueType, view.value, nextEntry.value)) {
        JST_WARN("[NEXUS] Failed to decode manifest entry '{}' with type '{}'.", keyString, view.valueType);
        return 0;
    }

    std::lock_guard<std::mutex> guard(manifestMutex);
    auto& entries = stagingManifestCache[keyString];
    const auto existing = std::find_if(entries.begin(), entries.end(), [&nextEntry](const ManifestEntry& entry) {
        return entry.start == nextEntry.start && entry.end == nextEntry.end;
    });

    if (existing != entries.end()) {
        *existing = std::move(nextEntry);
        return 1;
    }

    entries.push_back(std::move(nextEntry));
    return 1;
}

void Nexus::Impl::commitManifestUpdate(const bool connected) {
    {
        std::lock_guard<std::mutex> guard(manifestMutex);
        manifestCache = std::move(stagingManifestCache);
        stagingManifestCache.clear();
    }

    {
        std::lock_guard<std::mutex> guard(stateMutex);
        bridgeConnected = connected;
        initialManifestReceived = true;
    }
}

bool Nexus::Impl::popMetric() {
    std::lock_guard<std::mutex> guard(metricsMutex);
    if (metricQueue.empty()) {
        return false;
    }

    currentMetric = std::move(metricQueue.front());
    metricQueue.pop_front();
    return true;
}

bool Nexus::Impl::popStatus() {
    std::lock_guard<std::mutex> guard(statusMutex);
    if (statusQueue.empty()) {
        return false;
    }

    currentStatus = std::move(statusQueue.front());
    statusQueue.pop_front();
    return true;
}

int Nexus::Impl::readMetric(StellineMetricView& view) {
    if (!popMetric()) {
        return 0;
    }

    view.block = currentMetric.block.c_str();
    view.key = currentMetric.key.c_str();
    view.label = currentMetric.label.c_str();
    view.format = currentMetric.format.c_str();
    view.value = currentMetric.value.c_str();
    return 1;
}

int Nexus::Impl::readStatus(StellineStatusView& view) {
    if (!popStatus()) {
        return 0;
    }

    view.status = currentStatus.status.c_str();
    view.log = currentStatus.log.c_str();
    return 1;
}

bool Nexus::Impl::decodeTypedValue(const std::string& type,
                                   const std::string& encoded,
                                   std::any& value) const {
    if (type == "string") {
        value = encoded;
        return true;
    }

    if (type == "i32") {
        return DecodeWithParser<Jetstream::I32>(encoded, value);
    }

    if (type == "u64") {
        return DecodeWithParser<Jetstream::U64>(encoded, value);
    }

    if (type == "f64") {
        return DecodeWithParser<Jetstream::F64>(encoded, value);
    }

    if (type == "vec<b64>") {
        // TODO: Implement base64 vector decoding.
        return true;
    }

    JST_WARN("[NEXUS] Unsupported manifest type '{}' storing it as a string.", type);
    value = encoded;
    return true;
}

bool Nexus::Available() {
    return Instance().pimpl->available();
}

bool Nexus::MetadataContains(const std::string& key, const Jetstream::U64 timestamp) {
    return Instance().pimpl->metadataContains(key, timestamp);
}

std::any Nexus::MetadataFetch(const std::string& key, const Jetstream::U64 timestamp) {
    return Instance().pimpl->metadataFetch(key, timestamp);
}

void Nexus::Initialize() {
    Instance().pimpl->initialize();
}

void Nexus::Deinitialize() {
    Instance().pimpl->deinitialize();
}

void Nexus::MetadataStore(const std::string& key,
                          const std::any& value,
                          const Jetstream::U64 start,
                          const Jetstream::U64 end) {
    Instance().pimpl->metadataStore(key, value, start, end);
}

void Nexus::MetadataClear() {
    Instance().pimpl->metadataClear();
}

Jetstream::U64 Nexus::MetadataSize() {
    return Instance().pimpl->metadataSize();
}

void Nexus::SignalStatus(const std::string& status, const std::string& log) {
    Instance().pimpl->signalStatus(status, log);
}

void Nexus::MetricsPush(const std::string& blockName, const MetricsList& metrics) {
    Instance().pimpl->metricsPush(blockName, metrics);
}

void Nexus::MetricsFlush() {
    Instance().pimpl->metricsFlush();
}

Nexus& Nexus::Instance() {
    static Nexus nexus;
    return nexus;
}

}  // namespace stelline

extern "C" __attribute__((visibility("default"))) int StellineNexusReset() {
    stelline::NexusBridge::impl().reset();
    return 1;
}

extern "C" __attribute__((visibility("default"))) int StellineNexusManifestBegin() {
    stelline::NexusBridge::impl().beginManifestUpdate();
    return 1;
}

extern "C" __attribute__((visibility("default"))) int StellineNexusManifestPush(const StellineManifestView* view) {
    if (!view) {
        return 0;
    }

    return stelline::NexusBridge::impl().pushManifestEntry(*view);
}

extern "C" __attribute__((visibility("default"))) int StellineNexusManifestCommit(int connected) {
    stelline::NexusBridge::impl().commitManifestUpdate(connected != 0);
    return 1;
}

extern "C" __attribute__((visibility("default"))) int StellineNexusMetricRead(StellineMetricView* view) {
    if (!view) {
        return 0;
    }

    return stelline::NexusBridge::impl().readMetric(*view);
}

extern "C" __attribute__((visibility("default"))) int StellineNexusStatusRead(StellineStatusView* view) {
    if (!view) {
        return 0;
    }

    return stelline::NexusBridge::impl().readStatus(*view);
}
