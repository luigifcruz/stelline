#ifndef STELLINE_FBH5_WRITER_BLOCK_HH
#define STELLINE_FBH5_WRITER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Fbh5Writer : public Block::Config {
    std::string filepath = "./file.fbh5";
    bool overwrite = false;
    bool recording = false;

    JST_BLOCK_TYPE(fbh5_writer);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, overwrite, recording);
    JST_BLOCK_DESCRIPTION(
        "FBH5 Writer",
        "Sink that writes pre-arranged tensors to FBH5.",
        "# FBH5 Writer\n"
        "The FBH5 Writer block writes pre-arranged filterbank tensors to an FBH5 file using "
        "GPUDirect Storage. Input tensors must be F32 shaped as [time, beams, channels, "
        "intermediate frequencies] and are appended to the file each compute cycle.\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the output FBH5 file.\n"
        "- **Overwrite**: Whether to overwrite the file if it already exists.\n"
        "- **Recording**: Start or stop recording to the file.\n\n"

        "## Useful For\n"
        "- Recording beamformed spectra for technosignature or pulsar searches.\n"
        "- Producing FBH5 files compatible with blimpy tooling.\n"
        "- Streaming GPU-resident spectra straight to disk.\n\n"

        "## Examples\n"
        "- Record spectra to FBH5:\n"
        "  Config: File Path='scan.fbh5', Overwrite=true, Recording=true\n"
        "  Input: F32[16, 1, 192, 1] -> Appended to the file each cycle.\n\n"

        "## Implementation\n"
        "Input Buffer -> FBH5 Module -> HDF5 GDS File\n"
        "1. Builds the filterbank header from the input tensor shape.\n"
        "2. Opens the FBH5 file with the GPUDirect Storage HDF5 driver.\n"
        "3. Registers the input buffer with CUfile and appends each tensor directly from GPU memory."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_FBH5_WRITER_BLOCK_HH
