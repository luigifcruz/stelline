#include <stelline/nexus.hh>

#include <jetstream/instance.hh>
#include <jetstream/logger.hh>
#include <jetstream/app.hh>

#include <atomic>
#include <cstdio>
#include <chrono>
#include <mutex>
#include <thread>

#ifdef __linux__
#include <sched.h>
#endif

#include <unistd.h>

using namespace Jetstream;

namespace stelline {

struct RuntimeState {
    std::atomic<Instance*> instance{nullptr};
    std::thread metricsThread;
    std::atomic<bool> metricsThreadRunning{false};

    std::mutex logPipeMutex;
    int logReadFd = -1;
    int logWriteFd = -1;
    int savedStdoutFd = -1;
    int savedStderrFd = -1;

    void initializeNexus();
    void deinitializeNexus();

    int openLogPipe();
    void closeLogPipe();
    bool beginLogCapture();
    void endLogCapture();
};

void RuntimeState::initializeNexus() {
    Nexus::Initialize();
    Nexus::SignalStatus("running");

    metricsThreadRunning = true;
    metricsThread = std::thread([&]() {
        while (metricsThreadRunning) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            Instance* currentInstance = instance.load();
            if (!currentInstance) {
                Nexus::MetricsFlush();
                continue;
            }

            std::unordered_map<std::string, std::shared_ptr<Flowgraph>> flowgraphs;
            if (currentInstance->flowgraphList(flowgraphs) != Result::SUCCESS) {
                Nexus::MetricsFlush();
                continue;
            }

            for (const auto& [_, flowgraph] : flowgraphs) {
                for (const auto& [blockName, block] : flowgraph->blockList()) {
                    if (!block || block->state() != Block::State::Created || !block->interface()) {
                        continue;
                    }

                    const auto& metrics = block->interface()->metrics();
                    if (metrics.empty()) {
                        continue;
                    }

                    Nexus::MetricsPush(blockName, metrics);
                }
            }

            Nexus::MetricsFlush();
        }
    });
}

void RuntimeState::deinitializeNexus() {
    metricsThreadRunning = false;
    if (metricsThread.joinable()) {
        metricsThread.join();
    }

    Nexus::SignalStatus("stopped");
    Nexus::Deinitialize();
    instance.store(nullptr);
}

int RuntimeState::openLogPipe() {
#ifdef __linux__
    std::lock_guard<std::mutex> guard(logPipeMutex);

    if (logReadFd >= 0) {
        return logReadFd;
    }

    int pipeFds[2] = {-1, -1};
    if (pipe(pipeFds) != 0) {
        return -1;
    }

    logReadFd = pipeFds[0];
    logWriteFd = pipeFds[1];
    return logReadFd;
#else
    return -1;
#endif
}

void RuntimeState::closeLogPipe() {
    std::lock_guard<std::mutex> guard(logPipeMutex);

    if (logReadFd >= 0) {
        close(logReadFd);
        logReadFd = -1;
    }

    if (logWriteFd >= 0) {
        close(logWriteFd);
        logWriteFd = -1;
    }
}

bool RuntimeState::beginLogCapture() {
#ifdef __linux__
    std::lock_guard<std::mutex> guard(logPipeMutex);

    if (logWriteFd < 0) {
        return true;
    }

    if (unshare(CLONE_FILES) != 0) {
        return false;
    }

    savedStdoutFd = dup(STDOUT_FILENO);
    savedStderrFd = dup(STDERR_FILENO);
    if (savedStdoutFd < 0 || savedStderrFd < 0) {
        if (savedStdoutFd >= 0) {
            close(savedStdoutFd);
            savedStdoutFd = -1;
        }

        if (savedStderrFd >= 0) {
            close(savedStderrFd);
            savedStderrFd = -1;
        }

        return false;
    }

    if (dup2(logWriteFd, STDOUT_FILENO) < 0 || dup2(logWriteFd, STDERR_FILENO) < 0) {
        dup2(savedStdoutFd, STDOUT_FILENO);
        dup2(savedStderrFd, STDERR_FILENO);
        close(savedStdoutFd);
        close(savedStderrFd);
        savedStdoutFd = -1;
        savedStderrFd = -1;
        return false;
    }

    setvbuf(stdout, nullptr, _IOLBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    return true;
#else
    return false;
#endif
}

void RuntimeState::endLogCapture() {
    std::lock_guard<std::mutex> guard(logPipeMutex);

    fflush(stdout);
    fflush(stderr);

    if (savedStdoutFd >= 0) {
        dup2(savedStdoutFd, STDOUT_FILENO);
        close(savedStdoutFd);
        savedStdoutFd = -1;
    }

    if (savedStderrFd >= 0) {
        dup2(savedStderrFd, STDERR_FILENO);
        close(savedStderrFd);
        savedStderrFd = -1;
    }
}

}  // namespace stelline

static stelline::RuntimeState state;

extern "C" __attribute__((visibility("default"))) void CyberEtherPluginCreate(Instance* instance) {
    state.instance.store(instance);
    state.initializeNexus();
}

extern "C" __attribute__((visibility("default"))) void CyberEtherPluginDestroy(Instance*) {
    state.deinitializeNexus();
}

extern "C" __attribute__((visibility("default"))) int StellineLogPipeOpen() {
    return state.openLogPipe();
}

extern "C" __attribute__((visibility("default"))) void StellineLogPipeClose() {
    state.closeLogPipe();
}

extern "C" __attribute__((visibility("default"))) int StellineRunApp(int argc, char* argv[]) {
    state.beginLogCapture();
    const auto res = RunAppNative(argc, argv, CyberEtherPluginCreate, CyberEtherPluginDestroy);
    state.endLogCapture();
    return res;
}
