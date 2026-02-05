#include <stelline/system_info.hh>

#include <holoscan/logger/logger.hpp>

namespace stelline {

SystemInfo& SystemInfo::instance() {
    static SystemInfo inst;
    return inst;
}

SystemInfo::SystemInfo()
    : _configured(false)
    , _systemName("Unknown")
    , _unifiedMemory(false)
    , _discreteGpu(true) {}

void SystemInfo::configure(const std::string& systemName, bool unifiedMemory, bool discreteGpu) {
    if (_configured) {
        HOLOSCAN_LOG_WARN("SystemInfo already configured, ignoring reconfiguration.");
        return;
    }
    _systemName = systemName;
    _unifiedMemory = unifiedMemory;
    _discreteGpu = discreteGpu;
    _configured = true;
}

const std::string& SystemInfo::systemName() const {
    return _systemName;
}

const bool& SystemInfo::unifiedMemory() const {
    return _unifiedMemory;
}

const bool& SystemInfo::discreteGpu() const {
    return _discreteGpu;
}

}  // namespace stelline
