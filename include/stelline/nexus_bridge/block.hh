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
        "# Nexus Bridge\n"
        "The Nexus Bridge block streams Nexus metadata into the flowgraph environment. It "
        "subscribes to the Nexus deployment at the configured URL and mirrors every key/value "
        "entry into the environment without blocking compute. The block requires the CPU "
        "device and the Python runtime.\n\n"

        "## Arguments\n"
        "- **Nexus URL**: Address of the Nexus metadata source.\n\n"

        "## Useful For\n"
        "- Feeding observatory and observation metadata to sink blocks like the UVH5 Writer.\n"
        "- Keeping the flowgraph environment synchronized with live telescope state.\n"
        "- Monitoring bridge connectivity and the number of mirrored variables.\n\n"

        "## Examples\n"
        "- Mirror Nexus metadata:\n"
        "  Config: Nexus URL='https://nexus.stelline.space'\n"
        "  Environment keys like 'observatory.name' become available to other blocks.\n\n"

        "## Implementation\n"
        "Nexus Subscription -> Watcher Thread -> Flowgraph Environment\n"
        "1. A background watcher subscribes to the Nexus metadata query.\n"
        "2. Metadata deltas are queued and applied to the flowgraph environment each throttled cycle.\n"
        "3. Connection status and variable count are published under the 'nexus.bridge' key."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_NEXUS_BRIDGE_BLOCK_HH
