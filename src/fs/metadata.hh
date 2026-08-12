#pragma once

#include <array>
#include <string>
#include <jetstream/memory/types.hh>
#include <jetstream/parser.hh>

namespace Jetstream::Modules {

struct ObservationBand {
    F64 frequency_start;
    F64 frequency_stop;
    U64 channel_start;
    U64 channel_stop;
    
    JST_SERDES(frequency_start, frequency_stop, channel_start, channel_stop);
};

struct ObservationTuning {
    std::string name;
    std::vector<ObservationBand> bands;

    JST_SERDES(name, bands);
};

struct AntennaCoordinates {
    F64 x;
    F64 y;
    F64 z;

    JST_SERDES(x, y, z);
};

struct AntennaPointing {
    F64 ra;
    F64 dec;
    std::string source_name;

    JST_SERDES(ra, dec, source_name);
};

struct AntennaDetails {
    std::string name;
    U64 number;
    F32 diameter;
    AntennaCoordinates position;
    AntennaPointing pointing;
    std::vector<ObservationTuning> tunings;

    JST_SERDES(name, number, diameter, position, pointing, tunings);
};

struct ObservationFengine {
    // U64 synctime;
    F64 sample_period;
    
    JST_SERDES(/* synctime, */sample_period);
};

struct ObservationIers {
    // F64 pm_x_arcsec;
    // F64 pm_y_arcsec;
    F64 ut1_utc;
    
    JST_SERDES(/* pm_x_arcsec, pm_y_arcsec, */ut1_utc);
};

struct TelescopeCoordinates {
    F64 latitude;
    F64 longitude;
    F32 altitude;

    JST_SERDES(latitude, longitude, altitude);
};

struct TelescopeInfo {
    std::string name;
    TelescopeCoordinates coordinates;
    std::vector<AntennaDetails> antennas;
    ObservationIers iers;

    JST_SERDES(name, coordinates, antennas, iers);
};

}