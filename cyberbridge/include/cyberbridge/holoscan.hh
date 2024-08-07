#ifndef CYBERBRIDGE_HOLOSCAN_HH
#define CYBERBRIDGE_HOLOSCAN_HH

#include <functional>
#include <memory>
#include <string>

#include <jetstream/base.hh>
#include <holoscan/holoscan.hpp>

using namespace Jetstream;

namespace CyberBridge {

class Holoscan {
 public:
    Holoscan(Holoscan const&) = delete;
    void operator=(Holoscan const&) = delete;

    typedef std::function<std::shared_ptr<holoscan::Application>()> AppFactory;

    static Holoscan& GetInstance();

    static bool IsAppRunning() {
        return GetInstance().isAppRunning();
    }

    static holoscan::Application& GetApp() {
        return GetInstance().getApp();
    }

    static Result RegisterApp(const AppFactory& factory) {
        return GetInstance().registerApp(factory);
    }

    static Result RegisterTap(const std::string& name, std::function<Result()> tap) {
        return GetInstance().registerTap(name, tap);
    }

    static Result RemoveTap(const std::string& name) {
        return GetInstance().removeTap(name);
    }

    static Result LockContext(std::function<Result()> callback) {
        return GetInstance().lockContext(callback);
    }

    static Result StartApp() {
        return GetInstance().startApp();
    }

    static Result StopApp() {
        return GetInstance().stopApp();
    }

    static Result StartRender(const std::string& flowgraphPath = "",
                             const Backend::Config& backendConfig = {},
                             const Viewport::Config& viewportConfig = {},
                             const std::function<Result()>& callback = nullptr);

    template<Device D, typename T>
    static std::shared_ptr<holoscan::Tensor> TensorToHoloscan(Tensor<D, T>& tensor);

 private:
    Holoscan();
    ~Holoscan();

    struct Impl;
    std::unique_ptr<Impl> pimpl;

    bool isAppRunning();
    holoscan::Application& getApp();
    Result registerApp(const AppFactory& factory);
    Result registerTap(const std::string& name, std::function<Result()> tap);
    Result removeTap(const std::string& name);
    Result lockContext(std::function<Result()> callback);
    Result startApp();
    Result stopApp();
};

}  // namespace CyberBridge

#endif  // CYBERBRIDGE_HOLOSCAN_HH