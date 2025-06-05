#include <holoscan/holoscan.hpp>

#include <stelline/types.hh>
#include <stelline/yaml/types/pipeline.hh>

using namespace holoscan;
using namespace stelline;

#ifdef STELLINE_BIT_SOCKET
#include <stelline/bits/socket/base.hh>
#endif
#ifdef STELLINE_BIT_TRANSPORT
#include <stelline/bits/transport/base.hh>
#endif
#ifdef STELLINE_BIT_FRBNN
#include <stelline/bits/frbnn/base.hh>
#endif
#ifdef STELLINE_BIT_BLADE
#include <stelline/bits/blade/base.hh>
#endif
#ifdef STELLINE_BIT_FILESYSTEM
#include <stelline/bits/filesystem/base.hh>
#endif

#ifdef STELLINE_BIT_SOCKET
using namespace stelline::bits::socket;
#endif
#ifdef STELLINE_BIT_TRANSPORT
using namespace stelline::bits::transport;
#endif
#ifdef STELLINE_BIT_FRBNN
using namespace stelline::bits::frbnn;
#endif
#ifdef STELLINE_BIT_BLADE
using namespace stelline::bits::blade;
#endif
#ifdef STELLINE_BIT_FILESYSTEM
using namespace stelline::bits::filesystem;
#endif

class DefaultOp : public holoscan::Application {
 public:
    void compose() override {
        std::shared_ptr<Resource> pool = make_resource<UnboundedAllocator>("pool");

        auto descriptor = FetchNode<PipelineDescriptor>(this, "stelline");
        HOLOSCAN_LOG_INFO("Pipeline Descriptor: {}", descriptor);

        std::unordered_map<std::string, BitInterface> map;
        std::vector<std::pair<std::string, std::string>> flows;

        for (const auto& node : descriptor.graph) {
#ifdef STELLINE_BIT_SOCKET
            if (node.bit == "socket_bit") {
                map[node.id] = SocketBit(this, pool, map.size(), node.configuration);
            }
#endif

#ifdef STELLINE_BIT_TRANSPORT
            if (node.bit == "transport_bit") {
                map[node.id] = TransportBit(this, pool, map.size(), node.configuration);
            }
#endif

#ifdef STELLINE_BIT_BLADE
            if (node.bit == "blade_bit") {
                map[node.id] = BladeBit(this, pool, map.size(), node.configuration);
            }
#endif

#ifdef STELLINE_BIT_FILESYSTEM
            if (node.bit == "filesystem_bit") {
                map[node.id] = FilesystemBit(this, pool, map.size(), node.configuration);
            }
#endif

#ifdef STELLINE_BIT_FRBNN
            if (node.bit == "frbnn_inference_bit") {
                map[node.id] = FrbnnInferenceBit(this, pool, map.size(), node.configuration);
            }

            if (node.bit == "frbnn_detection_bit") {
                map[node.id] = FrbnnDetectionBit(this, pool, map.size(), node.configuration);
            }
#endif

            if (!map.contains(node.id)) {
                HOLOSCAN_LOG_ERROR("Unknown bit: '{}'", node.bit);
                throw std::runtime_error("Unknown bit.");
            }

            for (const auto& [_, value] : node.inputMap) {
                flows.push_back({node.id, value});
            }
        }

        for (const auto& [dst, src] : flows) {
            if (!map.contains(src)) {
                HOLOSCAN_LOG_ERROR("Unknown node: '{}'", src);
                throw std::runtime_error("Unknown node.");
            }

            if (!map.contains(dst)) {
                HOLOSCAN_LOG_ERROR("Unknown node: '{}'", dst);
                throw std::runtime_error("Unknown node.");
            }

            add_flow(map[src].second, map[dst].first);
        }
    }
};

int main(int argc, char** argv) {
    // Check arguments.

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file_path>\n";
        std::cerr << "Options:\n";
        std::cerr << "  --help    Show this help message\n";
        return 1;
    }

    // Check help flag.

    if (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        std::cout << "Usage: " << argv[0] << " <config_file_path>\n";
        std::cout << "Options:\n";
        std::cout << "  --help    Show this help message\n";
        return 0;
    }

    // Check version flag.

    if (std::string(argv[1]) == "--version" || std::string(argv[1]) == "-v") {
        std::cout << "Stelline v" << STELLINE_VERSION_STR << " (build type: " << STELLINE_BUILD_TYPE << ")\n";
        return 0;
    }

    // Check configuration file.

    const std::string configurationFilePath = argv[1];
    std::cout << "Using configuration file: " << configurationFilePath << "\n";

    // Start application.

    auto app = holoscan::make_application<DefaultOp>();

    app->config(std::filesystem::path(configurationFilePath));
    app->run();

    return 0;
}
