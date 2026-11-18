#ifndef STELLINE_FS_RECORDING_TIMELINE_HH
#define STELLINE_FS_RECORDING_TIMELINE_HH

#include <optional>

#include <jetstream/logger.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

#include "../block_timing.hh"

namespace Jetstream::Modules {

struct RecordingGap {
    std::optional<BlockTiming> timing;
    U64 firstMissing = 0;
    U64 missingBlocks = 0;
};

// A file starts at the first successfully written block. Once started, its
// block duration and phase are fixed; a discontinuity must not move the grid.
class RecordingTimeline {
 public:
    Result plan(const Tensor& input, RecordingGap& gap) const {
        gap = {};
        if (input.hasAttribute("duration")) {
            gap.timing = ReadBlockTiming(input);
            if (!gap.timing) {
                JST_ERROR("[RECORDING_TIMELINE] Expected a valid U64 timestamp and positive U64 duration.");
                return Result::ERROR;
            }
        }
        if (!started) {
            return Result::SUCCESS;
        }
        if (gap.timing.has_value() != (duration != 0)) {
            JST_ERROR("[RECORDING_TIMELINE] RF timing mode changed; start a new recording.");
            return Result::ERROR;
        }
        if (!gap.timing) {
            return Result::SUCCESS;
        }

        const auto [timestamp, span] = *gap.timing;
        if (span != duration || timestamp < nextTimestamp ||
            (timestamp - nextTimestamp) % duration != 0) {
            JST_ERROR("[RECORDING_TIMELINE] RF block [{}, +{}) is off the recording grid (expected {}, +{}); start a new recording.",
                      timestamp, span, nextTimestamp, duration);
            return Result::ERROR;
        }

        gap.firstMissing = nextTimestamp;
        gap.missingBlocks = (timestamp - nextTimestamp) / duration;
        if (gap.missingBlocks > kMaxGapBlocks) {
            JST_ERROR("[RECORDING_TIMELINE] RF gap of {} blocks exceeds the padding limit of {}; start a new recording.",
                      gap.missingBlocks, kMaxGapBlocks);
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    void commit(const RecordingGap& gap) {
        started = true;
        if (gap.timing) {
            duration = gap.timing->duration;
            nextTimestamp = gap.timing->timestamp + duration;
        }
    }

 private:
    // A bad forward jump must not turn into an unbounded series of writes.
    static constexpr U64 kMaxGapBlocks = 1024;
    bool started = false;
    U64 duration = 0;
    U64 nextTimestamp = 0;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_FS_RECORDING_TIMELINE_HH
