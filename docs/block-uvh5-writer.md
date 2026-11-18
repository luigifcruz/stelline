---
title: UVH5 Writer
description: Sink that writes pre-arranged correlation tensors to UVH5.
order: 42
category: Stelline
---

The UVH5 Writer is the recording sink of a correlator pipeline. It takes integrated visibility tensors from GPU memory and appends them to a [UVH5](https://pyuvdata.readthedocs.io/en/latest/uvh5_format.html) file, the HDF5-based interchange format used by pyuvdata and the wider radio interferometry toolchain. Writes go through the GPUDirect Storage HDF5 driver, so visibilities travel from GPU memory to NVMe without a host bounce.

## How it works

The block is idle until `recording` is enabled. At that point it assembles the complete UVH5 header from the flowgraph environment: telescope identity and location, the antenna table with positions and diameters, the baseline ordering, the frequency axis derived from the instance band, and the phase center. Recording refuses to start while any required key is missing, which prevents writing files with silent metadata gaps.

Each input tensor must carry a `timestamp` attribute. The writer uses the F-engine sync time and sample period from the environment to write the integration midpoint as a UTC Julian Date. It refreshes the time-dependent metadata (pointing and Earth orientation), computes the baseline UVW coordinates, and appends the integration. Closing the flowgraph finalizes the file.

Each input tensor must also carry `duration`, the number of F-engine samples it covers. The integration time is that duration multiplied by the F-engine sample period, and the first written integration establishes the grid. When a later integration reveals a gap, the writer inserts zero visibilities with flags set and zero sample weights at the missing times. It does not pad before the first or after the last integration. `/Header/extra_keywords` records the exact integer grid as `rf_start_sample` (the F-engine sample count where the first integration starts), `synctime`, and `rf_samples_per_row`. A change to the F-engine band or clock, a backwards or off-grid timestamp, change in duration, or gap larger than 1,024 integrations stops the recording before anything is appended and requires a new recording.

The frequency axis is derived rather than configured. The instance band defines the coarse channel range, and the input frequency dimension must be a whole multiple of that coarse channel count. The multiple is the number of fine channels per coarse channel from the upstream channelizer.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `filepath` | string | `./file.uvh5` | Path of the output file. The parent directory must exist. |
| `overwrite` | boolean | `false` | Replace an existing file instead of refusing to start. The existing file is kept until the first block of the new recording is written. |
| `recording` | boolean | `false` | Master switch for writing. |

## Input

| Name | Description |
|---|---|
| `input` | Contiguous `CF32` tensor shaped `[1, baselines, frequencies, 4]` with `timestamp` and `duration` attributes. |

The baseline count must match the antenna table, so `N` antennas require `N * (N + 1) / 2` baselines including autocorrelations. The polarization axis holds the four cross products of the `x` and `y` feeds. With the 28-antenna example environment the expected input is `CF32[1, 406, 192, 4]`.

## Attributes

| Attribute | Type | Direction | Description |
|---|---|---|---|
| `timestamp` | `U64` | Read from `input` | Required. First sample of the integration on the F-engine sample counter. |
| `duration` | `U64` | Read from `input` | Required, positive. Number of F-engine samples in the integration. Sets the integration time and gap padding grid. |

## Environment variables

The block is a pure consumer of the flowgraph environment. It reads the keys below, where `<name>` is an antenna name, `<tuning>` is the instance tuning, and `<band>` is the instance band index.

Read once when recording starts:

| Key | Purpose |
|---|---|
| `observatory.name` | Telescope name in the UVH5 header. |
| `observatory.coordinates.latitude` | Site latitude. |
| `observatory.coordinates.longitude` | Site longitude. |
| `observatory.coordinates.altitude` | Site altitude. |
| `observation.antennas.length` | Antenna count of the observation. |
| `observation.antennas.<index>` | Antenna name per index. Index `0` is also rechecked, as listed below. |
| `observatory.antenna.<name>.number` | Antenna number. |
| `observatory.antenna.<name>.diameter` | Dish diameter. |
| `observatory.antenna.<name>.position.x` | Antenna position, x axis. |
| `observatory.antenna.<name>.position.y` | Antenna position, y axis. |
| `observatory.antenna.<name>.position.z` | Antenna position, z axis. |

Read when recording starts and checked again before every input. A change stops the recording:

| Key | Purpose |
|---|---|
| `observation.antennas.0` | Antenna whose F-engine band and clock define the file axes. |
| `instance.bands.len` | Band count of this instance; must be one. |
| `instance.bands.0.tuning` | Tuning identifier. |
| `instance.bands.0.band_index` | Band index within the tuning. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.frequency_start` | Band start frequency in MHz. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.frequency_stop` | Band stop frequency in MHz. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.channel_start` | First coarse channel. |
| `observatory.antenna.<name>.tunings.<tuning>.bands.<band>.channel_stop` | Last coarse channel, exclusive. |
| `observatory.antenna.<name>.tunings.<tuning>.fengine.synctime` | F-engine sync time, the timestamp origin. |
| `observatory.antenna.<name>.tunings.<tuning>.fengine.sample_period` | F-engine sample period in seconds. |

Re-read per input timestamp:

| Key | Purpose |
|---|---|
| `observatory.antenna.<name>.pointing.ra` | Phase center right ascension in radians. |
| `observatory.antenna.<name>.pointing.dec` | Phase center declination in radians. |
| `observatory.antenna.<name>.pointing.source_name` | Phase center source name. |
| `observation.iers.pm_x_arcsec` | Polar motion, x component. |
| `observation.iers.pm_y_arcsec` | Polar motion, y component. |
| `observation.iers.ut1_utc` | The offset from UTC to UT1. |

The pointing keys are read from the first antenna of the observation. In a live deployment all of these come from the [Nexus Bridge](/docs/block-nexus-bridge).

## Metrics

The node reports the total data written, the current write bandwidth in MB/s, and the number of chunks committed to the file.

## Telemetry

The metrics below are reported to Nexus.

| Metric | Description |
|---|---|
| `bandwidth` | Write bandwidth in megabytes per second. |
| `totalDataWritten` | Total written UVH5 payload in megabytes. |
| `chunksWritten` | Total number of written UVH5 chunks, including padded integrations inserted for RF gaps. |

## Requirements

The GPUDirect Storage stack must be installed. See [host dependencies](/docs/host-dependencies) for the setup.
