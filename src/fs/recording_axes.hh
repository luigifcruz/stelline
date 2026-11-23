#pragma once

#include <optional>

#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct FrequencyAxis {
    F64 first = 0.0;
    F64 step = 0.0;
};

// The band spans [start, stop) over `coarseChannels` F-engine channels. Each
// coarse channel is split by a shifted FFT, which centers fine channel N / 2
// on the coarse channel center.
inline std::optional<FrequencyAxis> ChannelFrequencies(const F64 start,
                                                      const F64 stop,
                                                      const U64 coarseChannels,
                                                      const U64 channels) {
    if (coarseChannels == 0 || channels == 0 || channels % coarseChannels != 0 || stop == start) {
        return std::nullopt;
    }
    const U64 fineChannels = channels / coarseChannels;
    const F64 coarseWidth = (stop - start) / static_cast<F64>(coarseChannels);
    const F64 fineWidth = coarseWidth / static_cast<F64>(fineChannels);
    return FrequencyAxis{
        start + coarseWidth / 2.0 - static_cast<F64>(fineChannels / 2) * fineWidth,
        fineWidth,
    };
}

inline long double RfSampleUnixSeconds(const U64 synctime, const U64 sample, const F64 samplePeriod) {
    return static_cast<long double>(synctime) + static_cast<long double>(sample) * samplePeriod;
}

inline F64 UnixSecondsToMjd(const long double seconds) {
    constexpr long double kUnixEpochMjd = 40587.0L;
    return static_cast<F64>(kUnixEpochMjd + seconds / 86400.0L);
}

}  // namespace Jetstream::Modules
