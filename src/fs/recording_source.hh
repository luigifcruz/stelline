#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <string>
#include <vector>

#include <jetstream/flowgraph_environment.hh>
#include <jetstream/logger.hh>
#include <jetstream/parser.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct RecordingSource {
    std::string antenna;
    std::string tuning;
    std::vector<I64> bandIndices;
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

inline std::string NormalizeTuning(const std::string& tuning) {
    const auto first = tuning.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = tuning.find_last_not_of(" \t\n\r");
    std::string normalized = tuning.substr(first, last - first + 1);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return normalized;
}

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
    if (integer.value < 1) {
        JST_ERROR("{} Instance has no bands.", tag);
        return Result::ERROR;
    }
    const I64 bandCount = integer.value;
    if (!require("observation.antennas.0", text)) {
        return Result::INCOMPLETE;
    }
    source.antenna = text.value;

    struct Band {
        I64 index = 0;
        F64 frequencyStart = 0.0;
        F64 frequencyStop = 0.0;
        I64 channelStart = 0;
        I64 channelStop = 0;
    };
    std::vector<Band> bands;
    for (I64 slot = 0; slot < bandCount; ++slot) {
        const auto instanceBand = jst::fmt::format("instance.bands.{}", slot);
        if (!require(instanceBand + ".tuning", text)) {
            return Result::INCOMPLETE;
        }
        const auto tuning = NormalizeTuning(text.value);
        if (slot == 0) {
            source.tuning = tuning;
        } else if (tuning != source.tuning) {
            JST_ERROR("{} Instance bands use more than one tuning, but recording supports one tuning.", tag);
            return Result::ERROR;
        }
        Band entry;
        if (!require(instanceBand + ".band_index", integer)) {
            return Result::INCOMPLETE;
        }
        entry.index = integer.value;
        if (std::any_of(bands.begin(), bands.end(), [&](const Band& band) { return band.index == entry.index; })) {
            continue;
        }

        const auto band = jst::fmt::format("observatory.antenna.{}.tunings.{}.bands.{}",
                                           source.antenna, source.tuning, entry.index);
        if (!require(band + ".frequency_start", real)) {
            return Result::INCOMPLETE;
        }
        entry.frequencyStart = real.value;
        if (!require(band + ".frequency_stop", real)) {
            return Result::INCOMPLETE;
        }
        entry.frequencyStop = real.value;
        if (!require(band + ".channel_start", integer)) {
            return Result::INCOMPLETE;
        }
        entry.channelStart = integer.value;
        if (!require(band + ".channel_stop", integer)) {
            return Result::INCOMPLETE;
        }
        entry.channelStop = integer.value;

        if (entry.index < 0 || entry.channelStart < 0 || entry.channelStop <= entry.channelStart ||
            !std::isfinite(entry.frequencyStart) || !std::isfinite(entry.frequencyStop) ||
            entry.frequencyStart == entry.frequencyStop) {
            JST_ERROR("{} Invalid F-engine band metadata.", tag);
            return Result::ERROR;
        }
        bands.push_back(entry);
    }

    std::sort(bands.begin(), bands.end(), [](const Band& left, const Band& right) {
        return left.channelStart < right.channelStart;
    });
    const auto width = [](const Band& band) {
        return (band.frequencyStop - band.frequencyStart) / static_cast<F64>(band.channelStop - band.channelStart);
    };
    const F64 channelWidth = width(bands.front());
    for (size_t index = 1; index < bands.size(); ++index) {
        const Band& previous = bands[index - 1];
        const Band& band = bands[index];
        if (band.channelStart != previous.channelStop ||
            std::abs(width(band) - channelWidth) > 1e-9 * std::abs(channelWidth) ||
            std::abs(band.frequencyStart - previous.frequencyStop) > 1e-6 * std::abs(channelWidth)) {
            JST_ERROR("{} Instance bands {} and {} are not contiguous in channels and frequency.",
                      tag, previous.index, band.index);
            return Result::ERROR;
        }
    }
    for (const Band& band : bands) {
        source.bandIndices.push_back(band.index);
    }
    source.frequencyStart = bands.front().frequencyStart;
    source.frequencyStop = bands.back().frequencyStop;
    source.channelStart = bands.front().channelStart;
    source.channelStop = bands.back().channelStop;

    const auto tuning = jst::fmt::format("observatory.antenna.{}.tunings.{}", source.antenna, source.tuning);
    if (!require(tuning + ".fengine.synctime", integer)) {
        return Result::INCOMPLETE;
    }
    source.synctime = integer.value;
    if (!require(tuning + ".fengine.sample_period", real)) {
        return Result::INCOMPLETE;
    }
    source.samplePeriod = real.value;

    if (source.synctime < 0 || !std::isfinite(source.samplePeriod) || source.samplePeriod <= 0.0) {
        JST_ERROR("{} Invalid F-engine clock metadata.", tag);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

inline Result CheckRecordingSource(const Flowgraph::Environment& environment,
                                   const char* tag,
                                   const RecordingSource& expected) {
    RecordingSource current;
    if (LoadRecordingSource(environment, tag, current) != Result::SUCCESS || current != expected) {
        JST_ERROR("{} F-engine band or clock changed. Start a new recording.", tag);
        return Result::ERROR;
    }
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
