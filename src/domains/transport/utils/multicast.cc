#include <arpa/inet.h>

#include <dirent.h>
#include <errno.h>
#include <net/if.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <set>
#include <string>
#include <utility>

#include <jetstream/logger.hh>

#include "multicast.hh"

namespace stelline::domains::transport::utils {

struct MulticastMembership::Impl {
    int socketFd = -1;
    unsigned interfaceIndex = 0;
    std::string interfaceName;
    std::vector<U32> groups;
};

namespace {

bool IsIpv4Multicast(const U32 address) {
    return IN_MULTICAST(ntohl(address));
}

std::string FormatIpv4(const U32 address) {
    in_addr group = {};
    group.s_addr = address;

    char text[INET_ADDRSTRLEN] = {};
    if (::inet_ntop(AF_INET, &group, text, sizeof(text)) == nullptr) {
        return "<invalid>";
    }

    return text;
}

Result ResolveInterfaceName(const std::string& interfaceAddress, std::string& interfaceName) {
    interfaceName.clear();

    if (::if_nametoindex(interfaceAddress.c_str()) != 0) {
        interfaceName = interfaceAddress;
        return Result::SUCCESS;
    }

    const auto path = "/sys/bus/pci/devices/" + interfaceAddress + "/net";
    DIR* directory = ::opendir(path.c_str());
    if (directory == nullptr) {
        JST_ERROR("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Failed to resolve Linux interface name for '{}': {}.",
                  interfaceAddress,
                  std::strerror(errno));
        return Result::ERROR;
    }

    while (true) {
        const auto* entry = ::readdir(directory);
        if (entry == nullptr) {
            break;
        }

        const std::string candidate = entry->d_name;
        if (candidate == "." || candidate == "..") {
            continue;
        }

        if (!interfaceName.empty()) {
            ::closedir(directory);
            JST_ERROR("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] PCI device '{}' maps to multiple Linux interfaces.",
                      interfaceAddress);
            return Result::ERROR;
        }

        interfaceName = candidate;
    }

    ::closedir(directory);

    if (interfaceName.empty()) {
        JST_ERROR("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] No Linux interface found for PCI device '{}'.",
                  interfaceAddress);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace

MulticastMembership::MulticastMembership() {
    impl = std::make_unique<Impl>();
}

MulticastMembership::~MulticastMembership() {
    destroy();
    impl.reset();
}

Result MulticastMembership::create(const std::string& interfaceAddress,
                                   const std::vector<SubscriptionEndpoint>& subscriptions) {
    destroy();

    std::set<U32> uniqueGroups;
    for (const auto& subscription : subscriptions) {
        EndpointMatch destination;
        JST_CHECK(ParseEndpoint(subscription.destination, destination));

        if (!destination.hasIp || !IsIpv4Multicast(destination.ip)) {
            continue;
        }

        uniqueGroups.insert(destination.ip);
    }

    if (uniqueGroups.empty()) {
        return Result::SUCCESS;
    }

    JST_CHECK(ResolveInterfaceName(interfaceAddress, impl->interfaceName));

    impl->interfaceIndex = ::if_nametoindex(impl->interfaceName.c_str());
    if (impl->interfaceIndex == 0) {
        const auto errorText = std::string(std::strerror(errno));
        JST_ERROR("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Failed to resolve interface index for '{}': {}.",
                  impl->interfaceName,
                  errorText);
        impl->interfaceName.clear();
        return Result::ERROR;
    }

    const int membershipSocket = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (membershipSocket < 0) {
        const auto errorText = std::string(std::strerror(errno));
        JST_ERROR("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Failed to create multicast membership socket: {}.",
                  errorText);
        impl->interfaceIndex = 0;
        impl->interfaceName.clear();
        return Result::ERROR;
    }

    std::vector<U32> joinedGroups;
    joinedGroups.reserve(uniqueGroups.size());

    for (const auto groupAddress : uniqueGroups) {
        ip_mreqn request = {};
        request.imr_multiaddr.s_addr = groupAddress;
        request.imr_address.s_addr = INADDR_ANY;
        request.imr_ifindex = static_cast<int>(impl->interfaceIndex);

        if (::setsockopt(membershipSocket, IPPROTO_IP, IP_ADD_MEMBERSHIP, &request, sizeof(request)) != 0) {
            const auto errorText = std::string(std::strerror(errno));

            for (const auto joinedGroup : joinedGroups) {
                request.imr_multiaddr.s_addr = joinedGroup;
                ::setsockopt(membershipSocket, IPPROTO_IP, IP_DROP_MEMBERSHIP, &request, sizeof(request));
            }

            ::close(membershipSocket);

            JST_ERROR("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Failed to join multicast group {} on interface '{}': {}.",
                      FormatIpv4(groupAddress),
                      impl->interfaceName,
                      errorText);

            impl->interfaceIndex = 0;
            impl->interfaceName.clear();
            return Result::ERROR;
        }

        joinedGroups.push_back(groupAddress);
        JST_INFO("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Joined multicast group {} on interface '{}'.",
                 FormatIpv4(groupAddress),
                 impl->interfaceName);
    }

    impl->socketFd = membershipSocket;
    impl->groups = std::move(joinedGroups);

    return Result::SUCCESS;
}

void MulticastMembership::destroy() {
    if (impl->socketFd >= 0) {
        JST_INFO("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Leaving {} multicast groups on interface '{}'.",
                 impl->groups.size(),
                 impl->interfaceName);

        ip_mreqn request = {};
        request.imr_address.s_addr = INADDR_ANY;
        request.imr_ifindex = static_cast<int>(impl->interfaceIndex);

        for (const auto groupAddress : impl->groups) {
            request.imr_multiaddr.s_addr = groupAddress;

            if (::setsockopt(impl->socketFd, IPPROTO_IP, IP_DROP_MEMBERSHIP, &request, sizeof(request)) != 0) {
                JST_WARN("[TRANSPORT_UTILS_MULTICAST_MEMBERSHIP] Failed to leave multicast group {} on interface '{}': {}.",
                         FormatIpv4(groupAddress),
                         impl->interfaceName,
                         std::strerror(errno));
            }
        }

        ::close(impl->socketFd);
    }

    impl->socketFd = -1;
    impl->interfaceIndex = 0;
    impl->interfaceName.clear();
    impl->groups.clear();
}

}  // namespace stelline::domains::transport::utils
