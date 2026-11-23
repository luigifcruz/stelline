#ifndef STELLINE_UVH5_READER_BLOCK_HH
#define STELLINE_UVH5_READER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Uvh5Reader : public Block::Config {
    std::string filepath = "./file.uvh5";
    U64 batchSize = 1;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(stelline_uvh5_reader);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "UVH5 Reader",
        "Reads visibilities from a UVH5 file.",
        "# UVH5 Reader\n"
        "\n"
        "Reads visibilities from a UVH5 file, a few integrations per buffer, looping "
        "back at the end when asked. It usually starts an offline chain, replaying a "
        "UVH5 Writer recording into views or further processing.\n"
        "\n"
        "- **Playback is not paced.** A new buffer is read every cycle, as fast as the "
        "disk allows.\n"
        "- **Any setting change restarts the file.** Editing a setting, Playing "
        "included, reopens it at the first integration.\n"
        "- **Flags and sample counts are not read.** Integrations the UVH5 Writer "
        "padded come out as plain zeros.\n"
        "- **The header goes to the environment.** It replaces the `observatory` key "
        "with the telescope, antennas, and channels.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Output** (out) | CF32, CI32, CF64, or CI64, as stored | Batch Size "
        "integrations x baselines x channels x polarizations, with a `sampleAxis` on "
        "the integrations and a `channelAxis` on the channels. Sets no rate, frequency,"
        " or timing attributes. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | file.uvh5 in the working folder | An existing UVH5 file | "
        "File to read, picked from .uvh5 and .h5 files. Rows must be ordered by time, "
        "then baseline. |\n"
        "| **Batch Size** | 1 | 1 or more, dividing the integrations in the file | "
        "Integrations per output buffer. A value above the file length is clamped to "
        "it. |\n"
        "| **Loop** | On | On, Off | Restarts at the first integration after the last. "
        "Off stops reading after one pass. |\n"
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
        "All rows assume a CF32 file of 100 integrations with 406 baselines, 192 "
        "channels, and 4 polarizations.\n"
        "\n"
        "| Goal | Settings | Output |\n"
        "|---|---|---|\n"
        "| Step through integrations | Defaults | 1 integration x 406 baselines x 192 "
        "channels x 4 polarizations per buffer, 100 buffers per pass |\n"
        "| Read in blocks | Batch Size 10 | 10 integrations x 406 baselines x 192 "
        "channels x 4 polarizations per buffer, 10 buffers per pass |\n"
        "| Load the whole file | Batch Size 100 | 100 integrations x 406 baselines x "
        "192 channels x 4 polarizations, read again every cycle |\n"
        "| Play once | Loop off | 100 buffers of 1 integration, then no more |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file path is empty or the file does not exist | "
        "File Path is not set, or the file was moved | Pick the file in File Path. |\n"
        "| The block reports that it cannot open the file | The file is not an HDF5 "
        "file, or GPUDirect Storage is not working | Pick a UVH5 file, or see Setup. |\n"
        "| The block reports an irregular baseline-times dimension | The file does not "
        "hold every baseline at every time | Rewrite the file with a full baseline set "
        "per integration. |\n"
        "| The block reports that Batch Size must be an integer factor of the "
        "integrations | The integrations in the file do not divide evenly by Batch Size"
        " | Pick a divisor of the count in the message. |\n"
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
        "Each cycle reads the next integrations of the visibility dataset during the "
        "compute step, so a slow disk slows the whole chain. The antenna table and "
        "phase center are published again after every read.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [UVH5 Reader guide](https://stelline.space/docs/block-uvh5-reader) "
        "covers the file layout and environment keys.\n"
        "- Recordings come from the **UVH5 Writer**.\n"
        "- Spectra files play back in the **FBH5 Reader**."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_UVH5_READER_BLOCK_HH
