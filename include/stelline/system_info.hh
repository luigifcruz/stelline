#ifndef STELLINE_SYSTEM_INFO_HH
#define STELLINE_SYSTEM_INFO_HH

#include <string>

#include <stelline/common.hh>

namespace stelline {

class STELLINE_API SystemInfo {
 public:
    static SystemInfo& instance();

    void configure(const std::string& systemName, bool unifiedMemory, bool discreteGpu);

    const std::string& systemName() const;
    const bool& unifiedMemory() const;
    const bool& discreteGpu() const;

 private:
    SystemInfo();

    SystemInfo(const SystemInfo&) = delete;
    SystemInfo& operator=(const SystemInfo&) = delete;

    bool _configured;
    std::string _systemName;
    bool _unifiedMemory;
    bool _discreteGpu;
};

}  // namespace stelline

#endif  // STELLINE_SYSTEM_INFO_HH
