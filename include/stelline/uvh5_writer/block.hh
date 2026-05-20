#ifndef STELLINE_UVH5_WRITER_BLOCK_HH
#define STELLINE_UVH5_WRITER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Uvh5Writer : public Block::Config {
    std::string filepath = "./file.uvh5";
    U64 dspChannelizationRate = 1;
    U64 dspIntegrationRate = 1;
    bool overwrite = false;
    bool recording = false;

    JST_BLOCK_TYPE(uvh5_writer);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, dspChannelizationRate, dspIntegrationRate, overwrite, recording);
    JST_BLOCK_DESCRIPTION(
        "UVH5 Writer",
        "Sink that writes pre-arranged correlation tensors to UVH5.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_UVH5_WRITER_BLOCK_HH
