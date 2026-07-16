#ifndef STELLINE_FBH5_READER_BLOCK_HH
#define STELLINE_FBH5_READER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Fbh5Reader : public Block::Config {
    std::string filepath = "./file.fbh5";
    U64 batchSize = 8192;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(fbh5_reader);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "FBH5 Reader",
        "Source that iterates through an FBH5 file.",
        "# FBH5 Reader\n"
        "The FBH5 Writer block writes pre-arranged filterbank tensors to an FBH5 file using "
        "GPUDirect Storage. Output tensor is F32 shaped as [time, beams, channels, "
        "intermediate frequencies].\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the output FBH5 file.\n"
        "- **Batch Size**: Number of samples to read per processing cycle.\n"
        "- **Loop**: Whether to loop back to the start when reaching the end of the file.\n"
        "- **Playing**: Start or stop reading from the file.\n\n"

        "## Useful For\n"
        "- Playing back previously recorded signal data.\n"
        "- Testing signal processing chains with known input data.\n"
        "- Offline analysis of captured signals.\n\n"

        "## Implementation\n"
        "Input Buffer -> FBH5 Module -> HDF5 GDS File\n"
        "1. Opens the specified file in binary read mode.\n"
        "2. Reads Batch Size samples of the specified data type per cycle.\n"
        "3. When end of file is reached, either loops back or yields."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_FBH5_READER_BLOCK_HH
