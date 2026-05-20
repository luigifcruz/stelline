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
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_FBH5_WRITER_BLOCK_HH
