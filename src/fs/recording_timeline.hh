#pragma once

#include <jetstream/logger.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

#include "../block_timing.hh"

namespace Jetstream::Modules {

struct RecordingGap {
    BlockTiming timing;
    U64 firstMissing = 0;
    U64 missingBlocks = 0;
};

class RecordingTimeline {
 public:
    Result plan(const Tensor& input, const char* tag, RecordingGap& gap) const {
        gap = {};
        const auto timing = ReadBlockTiming(input);
        if (!timing) {
            JST_ERROR("{} Recording requires U64 'timestamp' and positive U64 'duration' input attributes.", tag);
            return Result::ERROR;
        }
        gap.timing = *timing;
        if (duration == 0) {
            return Result::SUCCESS;
        }

        const auto [timestamp, span] = gap.timing;
        if (span != duration || timestamp < nextTimestamp ||
            (timestamp - nextTimestamp) % duration != 0) {
            JST_ERROR("{} The RF block [{}, +{}) is off the recording grid (expected {}, +{}). Start a new recording.",
                      tag, timestamp, span, nextTimestamp, duration);
            return Result::ERROR;
        }

        gap.firstMissing = nextTimestamp;
        gap.missingBlocks = (timestamp - nextTimestamp) / duration;
        if (gap.missingBlocks > kMaxGapBlocks) {
            JST_ERROR("{} The RF gap of {} blocks exceeds the padding limit of {}. Start a new recording.",
                      tag, gap.missingBlocks, kMaxGapBlocks);
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    void commit(const RecordingGap& gap) {
        duration = gap.timing.duration;
        nextTimestamp = gap.timing.timestamp + duration;
    }

 private:
    static constexpr U64 kMaxGapBlocks = 1024;
    U64 duration = 0;
    U64 nextTimestamp = 0;
};

}  // namespace Jetstream::Modules
