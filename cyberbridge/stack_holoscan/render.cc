#include <mutex>
#include <unordered_map>
#include <future>
#include <functional>
#include <memory>
#include <string>

#include "cyberbridge/holoscan.hh"
#include <jetstream/store.hh>
#include <jetstream/blocks/manifest.hh>

#include "source_block.hh"

using namespace Jetstream;
using namespace Jetstream::Blocks;

namespace CyberBridge {

Result Holoscan::StartRender(const std::string& flowgraphPath,
                             const Backend::Config& backendConfig,
                             const Viewport::Config& viewportConfig,
                             const std::function<Result()>& callback) {
    JST_DEBUG("[BRIDGE] Starting the UI.");

    Instance instance;

    // Configure the instance interface.

    Instance::Config config = {
        .preferredDevice = Device::Vulkan,
        .enableCompositor = true,
        .backendConfig = backendConfig,
        .viewportConfig = viewportConfig,
    };

    JST_CHECK(instance.build(config));

    // Configure compositor.

    instance.compositor().showStore(true)
                         .showFlowgraph(true);

    // Inject the block manifest list.

    Store::LoadBlocks([](Block::ConstructorManifest& constructorManifest, 
                         Block::MetadataManifest& metadataManifest) {
        JST_TRACE("[BLOCKS] Loading block manifest list.");

        JST_BLOCKS_MANIFEST(
            Jetstream::Blocks::Source,
            // TODO: Add `Blocks::Sink` here.
        )

        return Result::SUCCESS;
    });

    // Load flowgraph if provided.

    if (!flowgraphPath.empty()) {
        JST_CHECK(instance.flowgraph().create(flowgraphPath));
    }

    // Start render loop.

    JST_CHECK(instance.start());

    // Start the compute worker.

    auto computeWorker = std::thread([&]{
        while (instance.computing()) {
            JST_CHECK_THROW(instance.compute());
        }
    });

    // Start the graphical worker.

    auto graphicalWorker = std::thread([&]{
        while (instance.presenting()) {
            if (instance.begin() == Result::SKIP) {
                continue;
            }

            if (callback) {
                JST_CHECK_THROW(callback());
            }

            JST_CHECK_THROW(instance.present());
            if (instance.end() == Result::SKIP) {
                continue;
            }
        }
    });

    // Wait user to close the window.

    while (instance.viewport().keepRunning()) {
        instance.viewport().pollEvents();
    }

    // Stop the instance and wait for threads.

    JST_CHECK(instance.reset());
    JST_CHECK(instance.stop());

    if (computeWorker.joinable()) {
        computeWorker.join();
    }

    if (graphicalWorker.joinable()) {
        graphicalWorker.join();
    }

    // Destroy the instance.

    JST_CHECK(instance.destroy());

    // Stop Holoscan app.

    JST_CHECK(Holoscan::StopApp());

    // Destroy all backends.

    Backend::DestroyAll();

    return Result::SUCCESS;
}

}  // namespace CyberBridge