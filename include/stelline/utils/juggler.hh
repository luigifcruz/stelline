#ifndef STELLINE_UTILS_JUGGLER_HH
#define STELLINE_UTILS_JUGGLER_HH

#include <memory>
#include <vector>
#include <functional>

#include <stelline/common.hh>

namespace stelline {

/**
 * @class Juggler
 * @brief A class that manages a pool of shared pointers to objects of type T.
 * 
 * The Juggler class provides a way to reuse memory by maintaining a pool of shared pointers.
 * It allows objects to be allocated and deallocated efficiently, reducing the overhead of memory allocation.
 * The class provides methods to resize the pool, clear the pool, and retrieve a shared pointer from the pool.
 * Unused pointers are recycled, and unique pointers are added to the used list.
 * 
 * @tparam T The type of objects managed by the Juggler.
 */
template<typename T>
class Juggler {
 public:
    /**
     * @brief Default constructor.
     */
    Juggler() = default;

    /**
     * @brief Resizes the pool and initializes objects using a lambda.
     * 
     * @param size The new size of the pool.
     * @param initializer A lambda function that creates and returns a new shared pointer of type T.
     */
    void resize(const uint64_t& size, const std::function<std::shared_ptr<T>()>& initializer) {
        clear();
        pool.reserve(size);
        used.reserve(size);
        for (uint64_t i = 0; i < size; ++i) {
            pool.push_back(initializer());
        }
    }

    /**
     * @brief Clears the pool and used objects.
     */
    void clear() {
        pool.clear();
        used.clear();
    }

    /**
     * @brief Retrieves the available size of the pool.
     *
     * @return The available size of the pool.
     */
    uint64_t available() const {
        return pool.size();
    }

    /**
     * @brief Retrieves the used size of the pool.
     *
     * @return The used size of the pool.
     */
    uint64_t referenced() const {
        return used.size();
    }

    /**
     * @brief Retrieves a reusable object from the pool.
     * 
     * @return A shared pointer to the retrieved object, or nullptr if the pool is empty.
     */
    std::shared_ptr<T> get() {
        // Recycle unused pointers.

        for (auto it = used.begin(); it != used.end();) {
            if ((*it).unique()) {
                pool.push_back(*it);
                it = used.erase(it);
            } else {
                ++it;
            }
        }

        // Check if there are any pointers available.

        if (pool.empty()) {
            return nullptr;
        }

        // Get the pointer from the pool.

        auto ptr = pool.back();
        pool.pop_back();

        // Add the pointer to the used list.

        used.push_back(ptr);

        // Return the pointer to caller.

        return ptr;
    }

 private:
    std::vector<std::shared_ptr<T>> pool;
    std::vector<std::shared_ptr<T>> used;
};

}  // namespace stelline

#endif  // STELLINE_UTILS_JUGGLER_HH
