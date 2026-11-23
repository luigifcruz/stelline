#ifndef STELLINE_FBH5_READER_BLOCK_HH
#define STELLINE_FBH5_READER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Fbh5Reader : public Block::Config {
    std::string filepath = "./file.fbh5";
    U64 batchSize = 1;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(stelline_fbh5_reader);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "FBH5 Reader",
        "Reads power spectra and their mask from an FBH5 file.",
        "# FBH5 Reader\n"
        "\n"
        "Reads power spectra and their mask from an FBH5 file, a few rows per buffer, "
        "looping back at the end when asked. It usually starts an offline chain, "
        "replaying an FBH5 Writer recording into views or further processing.\n"
        "\n"
        "- **Playback is not paced.** A new buffer is read every cycle, as fast as the "
        "disk allows.\n"
        "- **Any setting change restarts the file.** Editing a setting, Playing "
        "included, reopens it at the first spectrum.\n"
        "- **The header goes to the environment.** It replaces the `observatory` key, "
        "with one band per channel.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | F32 or F64, as stored | Batch Size spectra x IFs x "
        "channels, with a `sampleAxis` on the spectra and a `channelAxis` on the "
        "channels. Sets no rate, frequency, or timing attributes. |\n"
        "| **Mask** (out) | U8 | Same shape and axes as Output. A value of 1 flags a "
        "sample, such as a zero the FBH5 Writer padded in. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | file.fbh5 in the working folder | An existing FBH5 file | "
        "File to read, picked from .fbh5 and .h5 files. |\n"
        "| **Batch Size** | 1 | 1 or more, dividing the spectra in the file | Spectra "
        "per output buffer. A value above the file length is clamped to it. |\n"
        "| **Loop** | On | On, Off | Restarts at the first spectrum after the last. Off"
        " stops reading after one pass. |\n"
        "| **Playing** | On | On, Off | Reads a new buffer every cycle. Off stops "
        "reading, and turning it back on starts over. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Position** | Progress through the current pass in percent. Reads Batch "
        "Size times too high when Batch Size is above 1. |\n"
        "| **Bandwidth** | Smoothed read rate in MB/s. Reads 0 while Playing is off. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume an F32 file of 1,024 spectra with 1 IF and 4,096 channels.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Step through spectra | Defaults | 1 spectrum x 1 IF x 4,096 channels per "
        "buffer, 1,024 buffers per pass |\n"
        "| Read in blocks | Batch Size 64 | 64 spectra x 1 IF x 4,096 channels per "
        "buffer, 1 MiB each, 16 buffers per pass |\n"
        "| Load the whole file | Batch Size 1,024 | 1,024 spectra x 1 IF x 4,096 "
        "channels, read again every cycle |\n"
        "| Play once | Loop off | 1,024 buffers of 1 spectrum, then no more |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file path is empty or the file does not exist | "
        "File Path is not set, or the file was moved | Pick the file in File Path. |\n"
        "| The block reports that it cannot open the file | The file is not an HDF5 "
        "file, or GPUDirect Storage is not working | Pick an FBH5 file, or see Setup. |\n"
        "| The block reports that Batch Size must be an integer factor of the spectra |"
        " The spectra in the file do not divide evenly by Batch Size | Pick a divisor "
        "of the count in the message. |\n"
        "| The block reports an unsupported number of bits | The data is stored as 8 or"
        " 16-bit values | Convert the file to 32 or 64-bit floats. |\n"
        "| Views stop updating while Playing shows On | Loop is off and the pass ended "
        "| Turn on Loop, or toggle Playing to read the file again. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| Reading on a CUDA device | GPUDirect Storage, working as in [Host "
        "Dependencies](https://stelline.space/docs/host-dependencies). |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each cycle reads the next rows of the data and mask datasets during the "
        "compute step, so a slow disk slows the whole chain. The header becomes one "
        "antenna named after the beam number, with a band centered on each channel "
        "frequency.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [FBH5 Reader guide](https://stelline.space/docs/block-fbh5-reader) "
        "covers the file layout and environment keys.\n"
        "- Recordings come from the **FBH5 Writer**.\n"
        "- Visibility files play back in the **UVH5 Reader**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_FBH5_READER_BLOCK_HH
