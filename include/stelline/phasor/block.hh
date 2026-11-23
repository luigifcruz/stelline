#ifndef STELLINE_PHASOR_BLOCK_HH
#define STELLINE_PHASOR_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Phasor : public Block::Config {
    F64 observationFrequencyHz = 1.4e9;
    F64 channelBandwidthHz = 1e6;
    F64 totalBandwidthHz = 1e6;
    U64 frequencyStartIndex = 0;

    U64 referenceAntennaIndex = 0;
    F64 arrayReferenceLongitude = -52.874986919708704*M_PI/180;
    F64 arrayReferenceLatitude = -40.00980599999271*M_PI/180;
    F64 arrayReferenceAltitude = 818.6837417125288;

    JST_BLOCK_TYPE(stelline_phasor);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(
        observationFrequencyHz,
        channelBandwidthHz,
        totalBandwidthHz,
        frequencyStartIndex,
        referenceAntennaIndex,
        arrayReferenceLongitude,
        arrayReferenceLatitude,
        arrayReferenceAltitude
    );
    JST_BLOCK_DESCRIPTION(
        "Phasor",
        "Calculates the phasors for beamforming.",
        "# Phasor\n"
        "\n"
        "Computes the geometric delay of every beam and antenna and the calibrated "
        "phasor that steers each channel and polarization. It usually feeds a "
        "Beamformer, through an Expand Dims that adds the sample axis the Beamformer "
        "expects.\n"
        "\n"
        "- **Beams are steered relative to the boresight.** A beam at the boresight "
        "gets zero delay and its calibrations unchanged.\n"
        "- **Channel phases use each channel's lower edge.** Channel zero starts half "
        "the Total Bandwidth below the Observation Frequency.\n"
        "- **Phasors follow the Julian Date.** Every cycle recomputes the delays from "
        "the current inputs.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Antenna Positions** (in) | F64 | Antennas x 3, the ECEF position of each "
        "antenna in meters. The buffer must be contiguous. |\n"
        "| **Antenna Calibrations** (in) | CF64 | Antennas x channels x polarizations, "
        "with as many antennas as the positions. The buffer must be contiguous. |\n"
        "| **Boresight Coordinates** (in) | F64 | Two values, right ascension and "
        "declination in radians. The buffer must be contiguous. |\n"
        "| **Beam Coordinates** (in) | F64 | Beams x 2, right ascension and declination"
        " in radians. The buffer must be contiguous. |\n"
        "| **Julian Date** (in) | F64 | One value, the UTC Julian date, such as 2451545"
        " for noon on 1 January 2000. The buffer must be contiguous. |\n"
        "| **DUT1** (in) | F64 | One value, UT1 minus UTC in seconds. The buffer must "
        "be contiguous. |\n"
        "| **Delays** (out) | F64 | Beams x antennas, in seconds, relative to the "
        "reference antenna and the boresight. |\n"
        "| **Phasors** (out) | CF32 | Beams x antennas x channels x polarizations, each"
        " a calibration times its steering phase. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Observation Frequency** | 1,400 MHz | Any | Center of the observed band. |\n"
        "| **Channel Bandwidth** | 1 MHz | Any | Width of one calibration channel. Must"
        " match the channels upstream. |\n"
        "| **Total Bandwidth** | 1 MHz | Any | Width of the whole band. Channel zero "
        "starts half of it below Observation Frequency. |\n"
        "| **Frequency Start Index** | 0 | 0 or more | Band channel of the first "
        "calibration channel, such as the channel in the ATA Receiver Offset Block. |\n"
        "| **Reference Antenna Index** | 0 | Below the antenna count | Antenna whose "
        "delay is zero. |\n"
        "| **Array Reference Longitude** | -0.923 rad | Any | Longitude of the array "
        "reference position, east positive. |\n"
        "| **Array Reference Latitude** | -0.698 rad | Any | Geodetic latitude of the "
        "array reference position. |\n"
        "| **Array Reference Altitude** | 818.684 m | Any | Height of the array "
        "reference position above the WGS84 ellipsoid. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows use calibrations of 28 antennas x 192 channels x 2 polarizations.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Steer four beams | Defaults | Beam Coordinates of 4 beams | 4 beams x 28 "
        "antennas x 192 channels x 2 polarizations |\n"
        "| Feed a Beamformer | Expand Dims Axis 3 after the block | Beam Coordinates of"
        " 4 beams | 4 beams x 28 antennas x 192 channels x 1 sample x 2 polarizations |\n"
        "| Check the calibrations | Defaults | 1 beam at the Boresight Coordinates | "
        "Delays of 0 s, and phasors equal to the calibrations |\n"
        "| Phase a subband | Total Bandwidth 1,024 MHz, Frequency Start Index 352 | "
        "Beam Coordinates of 4 beams | Phases for 1,240 to 1,431 MHz |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The Beamformer reports that the phasor tensor must have 5 dimensions | "
        "Phasors have no sample axis | Insert an Expand Dims block with Axis 3 between "
        "them. |\n"
        "| The block reports that it expects input data types F64, CF64, F64, F64, F64,"
        " and F64 | An input has another type, such as CF32 calibrations | Convert it "
        "with a Cast block. |\n"
        "| The block reports a rank, size, or shape that does not match | An input "
        "breaks the shape in Ports, or the calibrations list another antenna count than"
        " the positions | Shape every input as in Ports. |\n"
        "| The block reports that the reference antenna index is out of range | "
        "Reference Antenna Index is not below the antenna count | Lower Reference "
        "Antenna Index. |\n"
        "| Beams point to the wrong place | Coordinates are in degrees or hours, or the"
        " Array Reference settings do not match the array | Give angles in radians, and"
        " set the Array Reference settings to the array site. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Astrometry from the ERFA library turns each right ascension and declination "
        "into an observed hour angle and declination at the Julian Date. Each delay is "
        "the travel time to an antenna along the beam direction, relative to the "
        "reference antenna and to the boresight. The model below starts from the hour "
        "angles and leaves out the astrometry.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def phasor(xyz, cal, boresight, beams, observation_frequency=1.4e9,\n"
        "           channel_bandwidth=1e6, total_bandwidth=1e6, frequency_start_index=0,\n"
        "           reference_antenna_index=0, array_reference_longitude=-0.923):\n"
        "    def delay(ha, dec):\n"
        "        h = ha - array_reference_longitude\n"
        "        s = [np.cos(dec) * np.cos(h), -np.cos(dec) * np.sin(h), np.sin(dec)]\n"
        "        return (xyz - xyz[reference_antenna_index]) @ s / 299792458.0\n"
        "    delays = np.array([delay(*b) - delay(*boresight) for b in beams])\n"
        "    channels = np.arange(cal.shape[1]) + frequency_start_index\n"
        "    freq = observation_frequency - total_bandwidth / 2 + channels * channel_bandwidth\n"
        "    steer = np.exp(-2j * np.pi * delays[:, :, None] * freq)\n"
        "    return delays, (steer[..., None] * cal).astype(np.complex64)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Phasor guide](https://stelline.space/docs/block-phasor) covers the "
        "delay pipeline and its frames.\n"
        "- **Beamformer** applies the phasors to antenna voltages.\n"
        "- **Expand Dims** adds the sample axis the Beamformer expects.\n"
        "- **Cast** converts calibrations to CF64."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_PHASOR_BLOCK_HH
