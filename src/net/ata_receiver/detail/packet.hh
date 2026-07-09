#ifndef STELLINE_ATA_RECEIVER_DETAIL_PACKET_HH
#define STELLINE_ATA_RECEIVER_DETAIL_PACKET_HH

#include <arpa/inet.h>

#include <jetstream/types.hh>

namespace Jetstream::Modules {

constexpr U64 kPacketDataSize = 6144;
constexpr U64 kPacketHeaderSize = 16;
constexpr U64 kPacketHeaderOffset = 42;

constexpr int kRxHeaderSegment = 0;
constexpr int kRxDataSegment = 1;

struct VoltagePacket {
    U8 version;
    U8 type;
    U16 numberOfChannels;
    U16 channelNumber;
    U16 antennaId;
    U64 timestamp;
    U8 data[];

    explicit VoltagePacket(const U8* pointer) {
        const auto* packet = reinterpret_cast<const VoltagePacket*>(pointer);
        version = packet->version;
        type = packet->type;
        numberOfChannels = ntohs(packet->numberOfChannels);
        channelNumber = ntohs(packet->channelNumber);
        antennaId = ntohs(packet->antennaId);
        timestamp = be64toh(packet->timestamp);
    }
} __attribute__((packed));

}  // namespace Jetstream::Modules

#endif  // STELLINE_ATA_RECEIVER_DETAIL_PACKET_HH
