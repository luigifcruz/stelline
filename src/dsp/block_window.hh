#ifndef STELLINE_DSP_BLOCK_WINDOW_HH
#define STELLINE_DSP_BLOCK_WINDOW_HH

#include <limits>

#include <jetstream/macros.hh>
#include <jetstream/memory/tensor.hh>
#include <jetstream/types.hh>

#include "../block_timing.hh"

namespace Jetstream::Modules {

struct WindowStep {
    bool accept = false;
    bool first = false;
    bool last = false;
    U64 index = 0;
};

// Multi-block timed windows start at exact multiples of their full RF span
// and require adjacent blocks of equal duration. Ignore old or overlapping
// arrivals before touching an open window; a forward gap abandons it.
class BlockWindow {
 public:
    void reset(const U64 blocksPerWindow) {
        *this = {};
        this->blocksPerWindow = blocksPerWindow;
    }

    WindowStep admit(const Tensor& input) {
        const auto timing = ReadBlockTiming(input);
        if (!timing) {
            if (input.hasAttribute("timestamp") || input.hasAttribute("duration")) {
                return {};
            }
            if (timed) {
                timed = false;
                filled = 0;
            }
            return advance();
        }

        if (timing->timestamp < nextTimestamp ||
            timing->duration > std::numeric_limits<U64>::max() / blocksPerWindow) {
            return {};
        }
        const U64 span = timing->duration * blocksPerWindow;
        const bool continues = timed && filled > 0 &&
                               timing->duration == blockDuration &&
                               timing->timestamp == nextTimestamp;
        timed = true;
        if (!continues) {
            filled = 0;
            // A single-block operation preserves the input interval as-is.
            if (span > std::numeric_limits<U64>::max() - timing->timestamp ||
                (blocksPerWindow > 1 && timing->timestamp % span != 0)) {
                return {};
            }
            windowStart = timing->timestamp;
            blockDuration = timing->duration;
        }

        nextTimestamp = timing->timestamp + timing->duration;
        return advance();
    }

    Result stamp(Tensor& output) const {
        if (!timed) {
            JST_CHECK(output.removeAttribute("timestamp"));
            return output.removeAttribute("duration");
        }

        JST_CHECK(output.setAttribute("timestamp", windowStart));
        JST_CHECK(output.setAttribute("duration", blockDuration * blocksPerWindow));
        return Result::SUCCESS;
    }

 private:
    WindowStep advance() {
        WindowStep step;
        step.accept = true;
        step.first = filled == 0;
        step.index = filled;
        filled += 1;
        step.last = filled == blocksPerWindow;
        if (step.last) {
            filled = 0;
        }
        return step;
    }

    U64 blocksPerWindow = 1;
    U64 filled = 0;
    bool timed = false;
    U64 windowStart = 0;
    U64 blockDuration = 0;
    // End of the last accepted RF block, even across window and mode changes.
    U64 nextTimestamp = 0;
};

}  // namespace Jetstream::Modules

#endif  // STELLINE_DSP_BLOCK_WINDOW_HH
