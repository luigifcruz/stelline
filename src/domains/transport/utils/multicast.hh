#pragma once

#include <memory>
#include <string>
#include <vector>

#include <jetstream/types.hh>

#include "endpoint.hh"

using namespace Jetstream;

namespace stelline::domains::transport::utils {

class MulticastMembership {
 public:
    MulticastMembership();
    MulticastMembership(const MulticastMembership&) = delete;
    MulticastMembership& operator=(const MulticastMembership&) = delete;
    ~MulticastMembership();

    Result create(const std::string& interfaceAddress,
                  const std::vector<SubscriptionEndpoint>& subscriptions);
    void destroy();

  private:
    struct Impl;

    std::unique_ptr<Impl> impl;
};

}  // namespace stelline::domains::transport::utils
