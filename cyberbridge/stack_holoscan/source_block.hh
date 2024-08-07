#ifndef CYBERBRIDGE_HOLOSCAN_SOURCE_BLOCK_HH
#define CYBERBRIDGE_HOLOSCAN_SOURCE_BLOCK_HH

#include "jetstream/memory/base.hh"

#include "source_module.hh"

namespace Jetstream::Blocks {

template<Device D, typename IT, typename OT>
class Source : public Block {
 public:
    // Configuration

    struct Config {
        std::string nodeName;
        std::string nodeOutputName;
        std::string shape = "8, 8192";

        JST_SERDES(nodeName, nodeOutputName, shape);
    };

    constexpr const Config& getConfig() const {
        return config;
    }

    // Input

    struct Input {
        JST_SERDES();
    };

    constexpr const Input& getInput() const {
        return input;
    }

    // Output

    struct Output {
        Tensor<D, IT> buffer;

        JST_SERDES(buffer);
    };

    constexpr const Output& getOutput() const {
        return output;
    }

    constexpr const Tensor<D, IT>& getOutputBuffer() const {
        return this->output.buffer;
    }

    // Housekeeping

    constexpr Device device() const {
        return D;
    }

    std::string id() const {
        return "holoscan-source";
    }

    std::string name() const {
        return "Holoscan Source";
    }

    std::string summary() const {
        return "Provides a source of data from Holoscan.";
    }

    std::string description() const {
        return "This block provides a source of data from Holoscan. "
               "It is used to connect to a Holoscan node and receive data from one of its outputs. "
               "First select the node and then the output you want to receive data from.\n"
               "\n"
               "## Suggestions\n"
               "  * To send data to Holoscan, use the *Holoscan Sink* block.\n"
               "\n"
               "## What is NVIDIA Holoscan?\n"
               "Holoscan SDK is an AI sensor processing platform that combines hardware systems for low-latency "
               "sensor and network connectivity, optimized libraries for data processing and AI, and core microservices "
               "to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build "
               "streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the "
               "Edge, Industrial Inspection and more. For more information, visit [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk).";
    }

    // Constructor

    Result create() {
        shape = parseShape(config.shape);

        JST_CHECK(instance().addModule(
            source, "source", {
                .nodeName = config.nodeName,
                .nodeOutputName = config.nodeOutputName,
                .shape = shape,
            }, {},
            locale()
        ));

        JST_CHECK(Block::LinkOutput("buffer", output.buffer, source->getOutputBuffer()));

        return Result::SUCCESS;
    }

    Result destroy() {
        JST_CHECK(instance().eraseModule(source->locale()));

        return Result::SUCCESS;
    }

    std::string warning() const {
        if (source) {
            return source->warning();
        }
        return "";
    }

    void drawControl() {
        if (!CyberBridge::Holoscan::IsAppRunning()) {
            ImGui::Text("Holoscan app is not running.");
            return;
        }

        auto& app = CyberBridge::Holoscan::GetApp();

        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("Node");
        ImGui::TableSetColumnIndex(1);
        ImGui::SetNextItemWidth(-1);
        static const char* noNodeMessage = "No nodes found";
        if (ImGui::BeginCombo("##NodeList", app.graph().is_empty() ? noNodeMessage : config.nodeName.c_str())) {
            for (const auto& node : app.graph().get_nodes()) {
                const bool isSelected = config.nodeName == node->name();
                if (ImGui::Selectable(node->name().c_str())) {
                    config.nodeName = node->name();
                }
                if (isSelected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        const auto& node = app.graph().find_node(config.nodeName);
        if (node) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Node Output");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            static const char* noNodeOutputMessage = "No node output found";
            if (ImGui::BeginCombo("##NodeOutputList", node->spec()->outputs().empty() ? noNodeOutputMessage : config.nodeOutputName.c_str())) {
                for (const auto& [name, spec] : node->spec()->outputs()) {
                    const bool isSelected = config.nodeOutputName == name;
                    if (ImGui::Selectable(name.c_str())) {
                        config.nodeOutputName = name;
                    }
                    if (isSelected) {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }

            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::Text("Shape");
            ImGui::TableSetColumnIndex(1);
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputText("##Shape", &config.shape, ImGuiInputTextFlags_EnterReturnsTrue)) {
                shape = parseShape(config.shape);
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Reconnecting..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }

        if (node && node->spec()->outputs().contains(config.nodeOutputName)) {
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            ImGui::TableSetColumnIndex(1);
            const F32 fullWidth = ImGui::GetContentRegionAvail().x;
            if (ImGui::Button("Connect Holoscan Tap", ImVec2(fullWidth, 0))) {
                JST_DISPATCH_ASYNC([&](){
                    ImGui::InsertNotification({ ImGuiToastType_Info, 1000, "Connecting..." });
                    JST_CHECK_NOTIFY(instance().reloadBlock(locale()));
                });
            }
        }
    }

    constexpr bool shouldDrawControl() const {
        return true;
    }

 private:
    JST_DEFINE_IO();

    std::shared_ptr<Jetstream::Source<D, IT>> source;
    std::vector<U64> shape;

    static std::vector<U64> parseShape(const std::string& shape) {
        std::vector<U64> result;
        std::istringstream ss(shape);
        std::string token;
        while (std::getline(ss, token, ',')) {
            result.push_back(std::stoull(token));
        }
        return result;
    }
};

}  // namespace Jetstream::Blocks

JST_BLOCK_ENABLE(Source, is_specialized<Jetstream::Source<D, IT>>::value &&
                         std::is_same<OT, void>::value)

#endif  // CYBERBRIDGE_HOLOSCAN_SOURCE_BLOCK_HH