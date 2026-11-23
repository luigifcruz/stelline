#ifndef STELLINE_FBH5_WRITER_BLOCK_HH
#define STELLINE_FBH5_WRITER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Fbh5Writer : public Block::Config {
    std::string filepath = "./file.fbh5";
    bool overwrite = false;
    bool recording = false;

    JST_BLOCK_TYPE(stelline_fbh5_writer);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, overwrite, recording);
    JST_BLOCK_DESCRIPTION(
        "FBH5 Writer",
        "Records power spectra to an FBH5 file.",
        "# FBH5 Writer\n"
        "\n"
        "Appends each incoming buffer of power spectra to an FBH5 file while Recording "
        "is on. It usually ends a beamformer chain, right after a Detector.\n"
        "\n"
        "- **Metadata comes from the environment.** Band, clock, and pointing keys are "
        "usually published by a Nexus Bridge.\n"
        "- **Gaps are padded with flagged zeros.** Missing buffers on the `timestamp` "
        "grid become zero rows with a mask of 1.\n"
        "- **Recording never appends.** Each start writes a new file, and replacing an "
        "old one needs Overwrite on.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32 | Spectra x beams x channels x IFs, with a `sampleAxis`"
        " on the spectra and a `channelAxis` on the channels. Spectra on another axis "
        "are moved to the front. Needs U64 `timestamp` and `duration` attributes. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **File Path** | file.fbh5 in the working folder | A path in an existing "
        "folder | File to write. An old file there stays until the first buffer of the "
        "new recording lands. |\n"
        "| **Overwrite** | Off | On, Off | Lets a recording replace a file that already"
        " exists. |\n"
        "| **Recording** | Off | On, Off | Writes every buffer while on. |\n"
        "\n"
        "## Readouts\n"
        "\n"
        "| Readout | Shows |\n"
        "|---|---|\n"
        "| **Bandwidth** | Smoothed write rate in MB/s. Reads 0 when not recording. |\n"
        "| **Total Data Written** | File size in MB, counting up while recording. Shows"
        " the file at File Path while stopped. |\n"
        "| **Chunks Written** | Buffers appended since recording started, padded gaps "
        "included. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume File Path is set, Recording is on, a band of 192 F-engine "
        "channels, and a `duration` of 8,192.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Record detected power | Defaults | 16 spectra x 1 beam x 192 channels x 1 IF"
        " | 16 rows of 192 channels per buffer, each row spanning 512 F-engine samples "
        "|\n"
        "| Keep fine channels | Defaults | 16 spectra x 1 beam x 1,536 channels x 1 IF "
        "| 16 rows of 1,536 channels, 8 per F-engine channel |\n"
        "| Record in Detector order | Defaults | 1 beam x 192 channels x 16 spectra x 1"
        " IF, `sampleAxis` 2 | The same rows as the first recipe, with the spectra "
        "moved to the front |\n"
        "| Survive dropped buffers | Defaults | 16 spectra per buffer at `timestamp` 0,"
        " 8,192, and 32,768 | 80 rows, with the 32 padded rows masked |\n"
        "| Replace an earlier take | Overwrite on | 16 spectra x 1 beam x 192 channels "
        "x 1 IF | The old file stays until the first buffer, then the new one takes its"
        " place |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the file already exists | A file sits at File Path "
        "and Overwrite is off, such as after restarting Recording | Turn on Overwrite, "
        "or pick a new File Path. |\n"
        "| The block reports a missing required environment value | The band, clock, or"
        " pointing keys are not published yet | Add a Nexus Bridge. The block starts "
        "once the keys arrive. |\n"
        "| The block reports that the input channel count does not split the F-engine "
        "channels | Channels is not a whole multiple of the F-engine channels in the "
        "band | Feed the whole band, split evenly into fine channels. |\n"
        "| The block reports that a buffer is off the recording grid, or that a gap "
        "exceeds the padding limit | A `timestamp` broke the grid, more than 1,024 "
        "buffers went missing, or the band or clock changed | Start a new recording "
        "with Overwrite on or a new File Path. |\n"
        "| The file holds scrambled spectra | The input has more than 1 beam, or more "
        "than 1 IF with several channels | Slice the input down to 1 beam and 1 IF. |\n"
        "\n"
        "## Setup\n"
        "\n"
        "| Platform | Requirement |\n"
        "|---|---|\n"
        "| GPUDirect Storage | The nvidia-fs driver and cuFile, working as in [Host "
        "Dependencies](https://stelline.space/docs/host-dependencies). |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer and its mask are appended through the GPUDirect Storage file "
        "driver during the compute step, so a slow disk slows the whole chain. The "
        "header is built when recording starts, and the start time and row duration "
        "come from the first buffer. A new recording goes to a staging file beside File"
        " Path and moves into place after its first buffer.\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [FBH5 Writer guide](https://stelline.space/docs/block-fbh5-writer) "
        "covers header fields and joining bands.\n"
        "- **Detector** produces the power spectra this block records.\n"
        "- Recordings play back in the **FBH5 Reader**.\n"
        "- **Nexus Bridge** publishes the band, clock, and pointing keys."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_FBH5_WRITER_BLOCK_HH
