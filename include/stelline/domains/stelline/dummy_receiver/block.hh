#ifndef STELLINE_DOMAINS_DUMMY_RECEIVER_BLOCK_HH
#define STELLINE_DOMAINS_DUMMY_RECEIVER_BLOCK_HH

#include <string>
#include <vector>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct DummyReceiver : public Block::Config {
    std::vector<U64> shape = {1, 1, 1024, 1};
    std::string dataType = "CF32";
    U64 period = 0;

    JST_BLOCK_TYPE(dummy_receiver);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(shape, dataType, period);
    JST_BLOCK_DESCRIPTION(
        "Dummy Receiver",
        "Generates fake tensor data for testing.",
        "// TODO: Write description."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_DOMAINS_DUMMY_RECEIVER_BLOCK_HH
