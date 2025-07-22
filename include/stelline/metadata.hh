#ifndef STELLINE_METADATA_HH
#define STELLINE_METADATA_HH

#include <string>
#include <vector>
#include <variant>
#include <memory>
#include <unordered_map>

#include <stelline/common.hh>

namespace stelline {

typedef std::variant<std::string, float> MetadataValueVariant;
typedef std::unordered_map<std::string, std::vector<std::tuple<std::string, std::string, MetadataValueVariant>>> MetadataStorage;
typedef std::shared_ptr<MetadataStorage> MetadataStoragePtr;

class STELLINE_API Metadata {
 public:
    Metadata() = default;
    ~Metadata() = default;

    const MetadataStoragePtr& metadata_storage() const;

    void load_metadata(const std::string& version, const MetadataStoragePtr& storage);
    void link_metadata(const MetadataStoragePtr& storage);
    void commit_metadata();

    std::vector<std::string> metadata_versions();
    std::vector<std::string> metadata_keys();
    std::vector<std::string> metadata_descriptions();

    bool metadata_contains(const std::string& key, const std::string& version = "");

    template<typename T>
    bool metadata_push(const std::string& key, const T& value, const std::string& description = "");

    template<typename T>
    bool metadata_pull(const std::string& key, T& value, const std::string& version = "");

 private:
    std::string version;
    MetadataStoragePtr local_storage;
    MetadataStoragePtr linked_storage;

    std::vector<std::string> versions;
    std::vector<std::string> keys;
    std::vector<std::string> descriptions;

    bool update_pending = false;
    void update_cached_lists();
};

}  // namespace stelline

#endif  // STELLINE_METADATA_HH
