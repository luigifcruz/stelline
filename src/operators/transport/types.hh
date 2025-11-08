#ifndef STELLINE_OPERATORS_TRANSPORT_TYPES_HH
#define STELLINE_OPERATORS_TRANSPORT_TYPES_HH

#include <arpa/inet.h>

#include <stelline/common.hh>

#define RX_HEADER 0
#define RX_DATA 1

namespace stelline::operators::transport {

struct VoltagePacket {
    uint8_t  version;
    uint8_t  type;
    uint16_t numberOfChannels;
    uint16_t channelNumber;
    uint16_t antennaId;
    uint64_t timestamp;
    uint8_t  data[];

    VoltagePacket(const uint8_t* ptr) {
        const auto* p = reinterpret_cast<const VoltagePacket*>(ptr);
        version = p->version;
        type = p->type;
        numberOfChannels = ntohs(p->numberOfChannels);
        channelNumber = ntohs(p->channelNumber);
        antennaId = ntohs(p->antennaId);
        timestamp = be64toh(p->timestamp);
    }
} __attribute__((packed));

}  // namespace stelline::operators::transport

#endif  // STELLINE_OPERATORS_TRANSPORT_TYPES_HH
