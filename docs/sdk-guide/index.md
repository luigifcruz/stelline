# SDK Guide

// TODO: Write.

- `operators`: Contains all the custom Holoscan operators (e.g., `transport`, `blade`, `frbnn`, `io`, etc.).
- `bits`: Contains glue code between the YAML configuration file and the custom Holoscan operators (e.g. `BladeBit`, `FrbnnInferenceBit`, `FrbnnDetectionBit`, `IoSinkBit`, `TransportBit`, etc.). Its purpose is to reduce the amount of boilerplate code needed to create custom pipelines.
- `recipes`: Contains the YAML configuration files that define the data processing pipelines without the need to write any glue code or compile any C++ code.
