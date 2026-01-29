#ifndef STELLINE_MANIFEST_HH
#define STELLINE_MANIFEST_HH

#include <any>
#include <memory>
#include <string>

#include <stelline/common.hh>

namespace stelline {

class STELLINE_API ManifestProvider {
 public:
    explicit ManifestProvider(const std::string& endpoint);
    ~ManifestProvider();

    std::any get(const std::string& key, uint64_t timestamp);

 private:
    struct Impl;
    std::unique_ptr<Impl> pimpl;
};

class STELLINE_API ManifestConsumer {
 public:
    virtual ~ManifestConsumer() = default;

    void setManifestProvider(ManifestProvider* provider);

 protected:
    ManifestProvider* manifest();

 private:
    ManifestProvider* manifest_ = nullptr;
};

}  // namespace stelline

#endif  // STELLINE_MANIFEST_HH
