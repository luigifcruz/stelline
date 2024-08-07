#include <mutex>
#include <unordered_map>
#include <future>
#include <functional>
#include <memory>
#include <string>

#include "cyberbridge/holoscan.hh"

using namespace Jetstream;

namespace CyberBridge {

struct Holoscan::Impl {
    std::mutex mtx;
    AppFactory factory;
    std::shared_ptr<holoscan::Application> app;
    std::future<Result> holoscanThreadHandler;
    std::unordered_map<std::string, std::function<Result()>> tapList;
};

Holoscan::Holoscan() {
    pimpl = std::make_unique<Impl>();
}

Holoscan::~Holoscan() {
    pimpl.reset();
}

Holoscan& Holoscan::GetInstance() {
    static Holoscan instance;
    return instance;
}

bool Holoscan::isAppRunning() {
    if (!pimpl->app) {
        return false;
    }

    if (!pimpl->holoscanThreadHandler.valid()) {
        return false;
    }

    if (pimpl->holoscanThreadHandler.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        return false;
    }

    return true;
}

holoscan::Application& Holoscan::getApp() {
    if (!pimpl->app) {
        JST_ERROR("[BRIDGE] Holoscan app is not registered.");
        JST_CHECK_THROW(Result::ERROR);
    }

    return *pimpl->app;
}

Result Holoscan::registerApp(const std::function<std::shared_ptr<holoscan::Application>()>& factory) {
    pimpl->factory = factory;
    pimpl->app = pimpl->factory();
    return Result::SUCCESS;
}

Result Holoscan::registerTap(const std::string& name, std::function<Result()> tap) {
    std::lock_guard<std::mutex> guard(pimpl->mtx);

    if (pimpl->tapList.contains(name)) {
        JST_ERROR("[BRIDGE] Tap '{}' already exists.", name);
        return Result::ERROR;
    }

    pimpl->tapList[name] = tap;

    Result result = Result::SUCCESS;

    if (isAppRunning()) {
        JST_CHECK(stopApp());

        for (const auto& [id, tap] : pimpl->tapList) {
            const auto& res = tap();

            if (res != Result::SUCCESS && id == name) {
                result = res;
            }
        }

        JST_CHECK(startApp());
    }

    if (result != Result::SUCCESS) {
        pimpl->tapList.erase(name);
    }

    return result;
}

Result Holoscan::removeTap(const std::string& name) {
    std::lock_guard<std::mutex> guard(pimpl->mtx);

    if (!pimpl->tapList.contains(name)) {
        JST_ERROR("[BRIDGE] Tap '{}' does not exist.", name);
        return Result::ERROR;
    }

    pimpl->tapList.erase(name);

    if (isAppRunning()) {
        JST_CHECK(stopApp());

        for (const auto& [name, tap] : pimpl->tapList) {
            JST_CHECK(tap());
        }

        JST_CHECK(startApp());
    }

    return Result::SUCCESS;
}

Result Holoscan::lockContext(std::function<Result()> callback) {
    if (!isAppRunning()) {
        JST_ERROR("[BRIDGE] Holoscan app is not running.");
        return Result::ERROR;
    }

    {
        std::lock_guard<std::mutex> guard(pimpl->mtx);
        JST_CHECK(callback());
    }

    return Result::SUCCESS;
}

Result Holoscan::startApp() {
    if (isAppRunning()) {
        JST_ERROR("[BRIDGE] Holoscan app is still running.");
        return Result::ERROR;
    }

    pimpl->holoscanThreadHandler = std::async(std::launch::async, [&](){
        JST_INFO("[BRIDGE] Starting Holoscan app.");

        try {
            std::future<void> future;
            future = pimpl->app->run_async();
            future.wait();
        } catch (const std::exception& e) {
            JST_ERROR("[BRIDGE] Holoscan app has crashed: {}", e.what());
            return Result::ERROR;
        }

        JST_INFO("[BRIDGE] Holoscan app has stopped.");
        return Result::SUCCESS;
    });

    // TODO: This is a hack to wait for the Holoscan executor to start.
    //       It would be nice to interface directly with GXF rather than
    //       these unreliable heuristics. But I can't find any public API
    //       to do so.

    
    for (const auto& node : pimpl->app->graph().get_nodes()) {
        while (true) {
            if (node->id() >= 0) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // TODO: This is still returning before all the nodes are scheduled.

    return Result::SUCCESS;
}

Result Holoscan::stopApp() {
    if (!isAppRunning()) {
        JST_ERROR("[BRIDGE] Holoscan app is not running.");
        return Result::ERROR;
    }

    JST_INFO("[BRIDGE] Stopping Holoscan app.");

    // TODO: This is a hack to wait for the app to stop. It wouldn't be necessary
    //       if the interrupt method returned an error code.

    while (isAppRunning()) {
        pimpl->app->executor().interrupt();
        pimpl->holoscanThreadHandler.wait_for(std::chrono::milliseconds(50));
    }
    
    // TODO: It would be nicer to keep the internal graph state and reload GXF
    //       instead of destroying and recreating it. But this would require
    //       modifying the Holoscan API.
    // 
    //       Something like `holoscan/core/fragment.cpp`:
    //       ```
    //       void Fragment::reload_context() {
    //           if (graph_) { graph_.reset(); }
    //           if (executor_) { executor_.reset(); }
    //           if (scheduler_) { scheduler_.reset(); }
    //           if (network_context_) { network_context_.reset(); }
    //       }
    //       ```
    //       
    //       Keeping this "hack" for now:
    pimpl->app.reset();
    pimpl->app = pimpl->factory();
    pimpl->app->compose_graph();

    return Result::SUCCESS;
}

}  // namespace CyberBridge