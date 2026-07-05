#ifndef STELLINE_NEXUS_BRIDGE_BLOCK_HH
#define STELLINE_NEXUS_BRIDGE_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct NexusBridge : public Block::Config {
    std::string url = "https://nexus.stelline.space";

    JST_BLOCK_TYPE(nexus_bridge);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(url);
    JST_BLOCK_DESCRIPTION(
        "Nexus Bridge",
        "Streams Nexus metadata into the flowgraph environment.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_NEXUS_BRIDGE_BLOCK_HH
