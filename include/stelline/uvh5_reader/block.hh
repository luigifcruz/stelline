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

    JST_BLOCK_TYPE(uvh5_reader);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "UVH5 Reader",
        "Source that iterates through an UVH5 file.",
        "# UVH5 Reader\n"
        "The UVH5 Reader block reads correlation tensors from an UVH5 file, using "
        "GPUDirect Storage on the CUDA device. Output tensor is complex and shaped "
        "as [time, baselines, channels, pol-products].\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the output UVH5 file.\n"
        "- **Batch Size**: Number of samples to read per processing cycle.\n"
        "- **Loop**: Whether to loop back to the start when reaching the end of the file.\n"
        "- **Playing**: Start or stop reading from the file.\n\n"

        "## Useful For\n"
        "- Playing back previously recorded signal data.\n"
        "- Testing signal processing chains with known input data.\n"
        "- Offline analysis of captured signals.\n\n"

        "## Implementation\n"
        "HDF5 (GDS) File -> UVH5 Module -> Output Buffer\n"
        "1. Opens the specified file in binary read mode.\n"
        "2. Reads Batch Size samples of the specified data type per cycle.\n"
        "3. When end of file is reached, either loops back or yields."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_UVH5_READER_BLOCK_HH
