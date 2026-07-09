#ifndef STELLINE_ATA_RECEIVER_DETAIL_GEOMETRY_HH
#define STELLINE_ATA_RECEIVER_DETAIL_GEOMETRY_HH

#include <jetstream/types.hh>

#include "kernel.hh"

namespace Jetstream::Modules {

inline U64 PacketSlotIndex(const AtaReceiverBlockGeometry& total,
                           const AtaReceiverBlockGeometry& partial,
                           const U64 antennaSlot,
                           const U64 channelSlot,
                           const U64 timeSlot) {
    const U64 channelSlots = total.numberOfChannels / partial.numberOfChannels;
    const U64 sampleSlots = total.numberOfSamples / partial.numberOfSamples;

    U64 index = 0;
    index += antennaSlot * channelSlots * sampleSlots;
    index += channelSlot * sampleSlots;
    index += timeSlot;
    return index;
}

inline U64 PacketElementOffset(const AtaReceiverBlockGeometry& total,
                               const AtaReceiverBlockGeometry& partial,
                               const U64 antennaSlot,
                               const U64 channelSlot,
                               const U64 timeSlot) {
    U64 offset = 0;
    offset += (antennaSlot * partial.numberOfAntennas) *
              total.numberOfChannels *
              total.numberOfSamples *
              total.numberOfPolarizations;
    offset += (channelSlot * partial.numberOfChannels) *
              total.numberOfSamples *
              total.numberOfPolarizations;
    offset += (timeSlot * partial.numberOfSamples) *
              total.numberOfPolarizations;
    return offset;
}

}  // namespace Jetstream::Modules

#endif  // STELLINE_ATA_RECEIVER_DETAIL_GEOMETRY_HH
