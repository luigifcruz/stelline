#pragma once

#include <cmath>
#include <string>

#include <jetstream/flowgraph_environment.hh>
#include <jetstream/logger.hh>
#include <jetstream/parser.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct RecordingSource {
    std::string antenna;
    std::string tuning;
    I64 bandIndex = 0;
    F64 frequencyStart = 0.0;
    F64 frequencyStop = 0.0;
    I64 channelStart = 0;
    I64 channelStop = 0;
    I64 synctime = 0;
    F64 samplePeriod = 0.0;

    U64 channelCount() const {
        return static_cast<U64>(channelStop - channelStart);
    }

    bool operator==(const RecordingSource&) const = default;
};

inline Result LoadRecordingSource(const Flowgraph::Environment& environment,
                                  const char* tag,
                                  RecordingSource& source) {
    struct {
        std::string value;
        JST_SERDES(value);
    } text;

    struct {
        F64 value = 0.0;
        JST_SERDES(value);
    } real;

    struct {
        I64 value = 0;
        JST_SERDES(value);
    } integer;

    const auto require = [&](const std::string& key, auto& value) {
        if (!environment.tryGet(key, value)) {
            JST_ERROR("{} Missing required '{}' environment value.", tag, key);
            return false;
        }
        return true;
    };

    source = {};
    if (!require("instance.bands.len", integer)) {
        return Result::INCOMPLETE;
    }
    if (integer.value != 1) {
        JST_ERROR("{} Recording supports exactly one band per instance, got {}.", tag, integer.value);
        return Result::ERROR;
    }
    if (!require("observation.antennas.0", text)) {
        return Result::INCOMPLETE;
    }
    source.antenna = text.value;
    if (!require("instance.bands.0.tuning", text)) {
        return Result::INCOMPLETE;
    }
    source.tuning = text.value;
    if (!require("instance.bands.0.band_index", integer)) {
        return Result::INCOMPLETE;
    }
    source.bandIndex = integer.value;

    const auto tuning = jst::fmt::format("observatory.antenna.{}.tunings.{}", source.antenna, source.tuning);
    const auto band = jst::fmt::format("{}.bands.{}", tuning, source.bandIndex);
    if (!require(band + ".frequency_start", real)) {
        return Result::INCOMPLETE;
    }
    source.frequencyStart = real.value;
    if (!require(band + ".frequency_stop", real)) {
        return Result::INCOMPLETE;
    }
    source.frequencyStop = real.value;
    if (!require(band + ".channel_start", integer)) {
        return Result::INCOMPLETE;
    }
    source.channelStart = integer.value;
    if (!require(band + ".channel_stop", integer)) {
        return Result::INCOMPLETE;
    }
    source.channelStop = integer.value;
    if (!require(tuning + ".fengine.synctime", integer)) {
        return Result::INCOMPLETE;
    }
    source.synctime = integer.value;
    if (!require(tuning + ".fengine.sample_period", real)) {
        return Result::INCOMPLETE;
    }
    source.samplePeriod = real.value;

    if (source.bandIndex < 0 || source.channelStart < 0 || source.channelStop <= source.channelStart ||
        !std::isfinite(source.frequencyStart) || !std::isfinite(source.frequencyStop) ||
        source.frequencyStart == source.frequencyStop || source.synctime < 0 ||
        !std::isfinite(source.samplePeriod) || source.samplePeriod <= 0.0) {
        JST_ERROR("{} Invalid F-engine band or clock metadata.", tag);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

inline Result CheckRecordingSource(const Flowgraph::Environment& environment,
                                   const char* tag,
                                   const RecordingSource& expected) {
    RecordingSource current;
    if (LoadRecordingSource(environment, tag, current) != Result::SUCCESS || current != expected) {
        JST_ERROR("{} F-engine band or clock changed; start a new recording.", tag);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
