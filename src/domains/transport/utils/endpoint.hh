#pragma once

#include <string>
#include <vector>
#include <arpa/inet.h>

#include <jetstream/types.hh>

using namespace Jetstream;

namespace stelline::domains::transport::utils {

struct EndpointMatch {
    bool hasIp = false;
    bool hasPort = false;
    in_addr_t ip = INADDR_ANY;
    U16 port = 0;
};

struct SubscriptionEndpoint {
    std::string source;
    std::string destination;
};

Result ParseEndpoint(const std::string& endpoint, EndpointMatch& match);

Result ParseSubscriptions(const std::string& text, std::vector<SubscriptionEndpoint>& subscriptions);

}  // namespace stelline::domains::transport::utils
