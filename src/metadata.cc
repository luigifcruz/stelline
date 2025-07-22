#include <stelline/metadata.hh>

namespace stelline {

void Metadata::load_metadata(const std::string& version, const MetadataStoragePtr& storage) {
    this->version = version;
    this->local_storage = storage;
    this->update_pending = true;
}

void Metadata::link_metadata(const MetadataStoragePtr& storage) {
    this->linked_storage = storage;
    this->update_pending = true;
}

void Metadata::commit_metadata() {
    if (this->linked_storage) {
        *this->linked_storage = *this->local_storage;
    }
}

const MetadataStoragePtr& Metadata::metadata_storage() const {
    return this->local_storage;
}

std::vector<std::string> Metadata::metadata_versions() {
    this->update_cached_lists();
    return this->versions;
}

std::vector<std::string> Metadata::metadata_keys() {
    this->update_cached_lists();
    return this->keys;
}

std::vector<std::string> Metadata::metadata_descriptions() {
    this->update_cached_lists();
    return this->descriptions;
}

bool Metadata::metadata_contains(const std::string& key, const std::string& version) {
    const auto& contains_key = (*this->local_storage).contains(key);
    if (version.empty() || !contains_key) {
        return contains_key;
    }
    for (const auto& [version, _1, _2] : (*this->local_storage).at(key)) {
        if (version == this->version) {
            return true;
        }
    }
    return false;
}

template<typename T>
bool Metadata::metadata_push(const std::string& key, const T& value, const std::string& description) {
    (*this->local_storage)[key].push_back({this->version, description, value});
    this->update_pending = true;
    return true;
}

template<typename T>
bool Metadata::metadata_pull(const std::string& key, T& value, const std::string& version) {
    if (!this->metadata_contains(key, version)) {
        throw std::out_of_range("Metadata key not found");
    }
    if (version.empty()) {
        const auto& [_1, _2, val] = (*this->local_storage).at(key).back();
        value = std::get<T>(val);
        return true;
    }
    for (const auto& [version, _1, val] : (*this->local_storage).at(key)) {
        if (version == this->version) {
            value = std::get<T>(val);
            return true;
        }
    }
    return false;
}

void Metadata::update_cached_lists() {
    // Check if update is pending.

    if (!this->update_pending) {
        return;
    }
    this->update_pending = false;

    // Clear existing lists.

    this->versions.clear();
    this->keys.clear();
    this->descriptions.clear();

    // Populate lists with latest metadata.

    for (const auto& [key, value] : (*this->local_storage)) {
        this->keys.push_back(key);

        for (const auto& [version, description, _] : value) {
            this->versions.push_back(version);
            this->descriptions.push_back(description);
        }
    }
}

template bool Metadata::metadata_push(const std::string&, const std::string&, const std::string&);
template bool Metadata::metadata_push(const std::string&, const float&, const std::string&);

template bool Metadata::metadata_pull(const std::string&, std::string&, const std::string&);
template bool Metadata::metadata_pull(const std::string&, float&, const std::string&);

}  // namespace stelline
