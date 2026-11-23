#ifndef STELLINE_POLARIZER_BLOCK_HH
#define STELLINE_POLARIZER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Polarizer : public Block::Config {
    std::string inputPolarization = "xy";
    std::string outputPolarization = "lr";
    U64 blockSize = 512;

    JST_BLOCK_TYPE(stelline_polarizer);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(inputPolarization, outputPolarization, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Polarizer",
        "Reorients the polarizations.",
        "# Polarizer\n"
        "\n"
        "Converts dual linear polarization voltages to left and right circular, or "
        "keeps one linear feed. It usually follows an ATA Receiver or a Beamformer, and"
        " feeds a Detector that forms power products.\n"
        "\n"
        "- **The input layout is fixed.** Axes are antennas, channels, samples, and "
        "polarizations, in that order.\n"
        "- **Circular output is not scaled.** Unpolarized power in each circular output"
        " is twice that of one feed.\n"
        "- **The output gains axis roles.** Channels and samples get `channelAxis` and "
        "`sampleAxis`, except when the input passes through.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Antennas x channels x samples x 2 polarizations, in "
        "a contiguous buffer. Any `batchAxis`, `channelAxis`, or `sampleAxis` must "
        "match that order. |\n"
        "| **Output** (out) | CF32 | Same shape, with 1 polarization for X and Y. Sets "
        "`channelAxis` and `sampleAxis`, and keeps every attribute. With XY, the input "
        "buffer passes through. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Input Polarization** | XY | XY | Basis of the input, X then Y along the "
        "polarization axis. |\n"
        "| **Output Polarizations** | LR | LR, XY, X, Y | With LR, left (X + jY) comes "
        "first and right (X - jY) second. With XY, the input passes through. With X or "
        "Y, only that feed is kept. |\n"
        "| **Block Size** | 512 threads | 1 to 1,024 threads | Threads per CUDA block. "
        "Ignored on CPU and with XY. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Circular from a linear feed | Defaults | 20 antennas x 192 channels x 8,192 "
        "samples x 2 polarizations | 20 antennas x 192 channels x 8,192 samples x 2 "
        "polarizations, left then right |\n"
        "| X feed only | Output Polarizations X | 20 antennas x 192 channels x 8,192 "
        "samples x 2 polarizations | 20 antennas x 192 channels x 8,192 samples x 1 "
        "polarization |\n"
        "| Y feed of one beam | Output Polarizations Y | 1 beam x 192 channels x 1,024 "
        "samples x 2 polarizations | 1 beam x 192 channels x 1,024 samples x 1 "
        "polarization |\n"
        "| Keep linear | Output Polarizations XY | 20 antennas x 192 channels x 8,192 "
        "samples x 2 polarizations | The same buffer, unchanged |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports an unsupported data type | The input is not CF32, such as "
        "CI8 from an ATA Receiver | Set the receiver Data Type to CF32, or convert with"
        " a Cast block. |\n"
        "| The block reports that the input must have 4 dimensions | The input has "
        "another rank | Reshape it to antennas x channels x samples x polarizations. |\n"
        "| The block reports that the polarization dimension must be 2 | The last axis "
        "holds one polarization, such as an ATA Receiver set to one polarization | Feed"
        " both polarizations. |\n"
        "| The block reports that `sampleAxis`, `channelAxis`, or `batchAxis` must be "
        "absent or at a set axis | The role sits on another axis | Move the roles with "
        "a Signal Axes block. |\n"
        "| Circular levels read 3 dB above one feed | Left and right are not scaled | "
        "Scale by 0.707 with a Multiply Constant block when levels must match. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Circular conversion turns the second feed a quarter turn in phase, then adds "
        "it to and subtracts it from the first feed, sample by sample. Single-feed "
        "output copies one polarization, and a matching basis hands the input on "
        "without a copy. The model below covers one buffer.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def polarize(x, input_polarization=\"xy\", output_polarization=\"lr\"):\n"
        "    if output_polarization == input_polarization:\n"
        "        return x\n"
        "    if output_polarization in (\"x\", \"y\"):\n"
        "        k = \"xy\".index(output_polarization)\n"
        "        return x[..., k:k + 1]\n"
        "    left = x[..., 0] + 1j * x[..., 1]\n"
        "    right = x[..., 0] - 1j * x[..., 1]\n"
        "    return np.stack([left, right], axis=-1)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Polarizer guide](https://stelline.space/docs/block-polarizer) covers "
        "the circular convention.\n"
        "- **Detector** forms power products from its output.\n"
        "- **Beamformer** combines antennas before or after it.\n"
        "- **Cast** converts CI8 voltages to CF32."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_POLARIZER_BLOCK_HH
