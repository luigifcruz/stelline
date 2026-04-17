#ifndef STELLINE_DOMAINS_FILESYSTEM_DUMMY_WRITER_BLOCK_HH
#define STELLINE_DOMAINS_FILESYSTEM_DUMMY_WRITER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct DummyWriter : public Block::Config {
    JST_BLOCK_TYPE(dummy_writer);
    JST_BLOCK_PARAMS();
    JST_BLOCK_DESCRIPTION(
        "Dummy Writer",
        "Sink that receives input tensors.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_DOMAINS_FILESYSTEM_DUMMY_WRITER_BLOCK_HH
