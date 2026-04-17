#include <sstream>
#include <stdexcept>

#include <jetstream/logger.hh>

#include "endpoint.hh"

namespace stelline::domains::transport::utils {

namespace {

std::string TrimWhitespace(const std::string& text) {
    const auto begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }

    const auto end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

}  // namespace

Result ParseEndpoint(const std::string& endpoint, EndpointMatch& match) {
    match = EndpointMatch{};

    const auto separator = endpoint.rfind(':');
    const bool hasPort = separator != std::string::npos;

    const std::string ipText = hasPort ? endpoint.substr(0, separator) : endpoint;
    const std::string portText = hasPort ? endpoint.substr(separator + 1) : "*";

    if (!ipText.empty() && ipText != "*") {
        in_addr addr = {};
        if (inet_pton(AF_INET, ipText.c_str(), &addr) != 1) {
            JST_ERROR("[TRANSPORT_UTILS] Invalid IPv4 endpoint: {}", endpoint);
            return Result::ERROR;
        }
        match.hasIp = true;
        match.ip = addr.s_addr;
    }

    if (!portText.empty() && portText != "*") {
        int port = 0;
        try {
            port = std::stoi(portText);
        } catch (const std::exception&) {
            JST_ERROR("[TRANSPORT_UTILS] Invalid UDP port in endpoint: {}", endpoint);
            return Result::ERROR;
        }

        if (port < 0 || port > 65535) {
            JST_ERROR("[TRANSPORT_UTILS] UDP port out of range in endpoint: {}", endpoint);
            return Result::ERROR;
        }

        match.hasPort = true;
        match.port = static_cast<U16>(port);
    }

    return Result::SUCCESS;
}

Result ParseSubscriptions(const std::string& text, std::vector<SubscriptionEndpoint>& subscriptions) {
    subscriptions.clear();
    std::istringstream stream(text);
    std::string line;

    while (std::getline(stream, line)) {
        const auto trimmed = TrimWhitespace(line);
        if (trimmed.empty()) {
            continue;
        }

        if (trimmed.rfind("- ", 0) != 0) {
            JST_ERROR("[TRANSPORT_UTILS] Each subscription line must start with '- '.");
            return Result::ERROR;
        }

        const auto body = TrimWhitespace(trimmed.substr(2));
        const auto arrow = body.find("->");
        if (arrow == std::string::npos) {
            JST_ERROR("[TRANSPORT_UTILS] Each subscription must use 'source:port -> destination:port'.");
            return Result::ERROR;
        }

        SubscriptionEndpoint subscription;
        subscription.source = TrimWhitespace(body.substr(0, arrow));
        subscription.destination = TrimWhitespace(body.substr(arrow + 2));
        if (subscription.source.empty() || subscription.destination.empty()) {
            JST_ERROR("[TRANSPORT_UTILS] Each subscription must define both source and destination.");
            return Result::ERROR;
        }

        subscriptions.push_back(std::move(subscription));
    }

    if (subscriptions.empty()) {
        JST_ERROR("[TRANSPORT_UTILS] No subscriptions were parsed from subscriptions config.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace stelline::domains::transport::utils
