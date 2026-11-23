#ifndef STELLINE_DETECTOR_BLOCK_HH
#define STELLINE_DETECTOR_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Detector : public Block::Config {
    U64 integrationRate = 1;
    U64 numberOfOutputPolarizations = 4;
    U64 blockSize = 512;

    JST_BLOCK_TYPE(stelline_detector);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(integrationRate, numberOfOutputPolarizations, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Detector",
        "Integrates detected power products.",
        "# Detector\n"
        "\n"
        "Turns dual-polarization voltages into power products and sums them over groups"
        " of consecutive samples. It usually follows a Beamformer and feeds an FBH5 "
        "Writer.\n"
        "\n"
        "- **The sample rate drops by Integration Rate.** The `sampleRate` attribute is"
        " divided by Integration Rate.\n"
        "- **Integration stays inside one buffer.** Every buffer must hold a multiple "
        "of Integration Rate samples.\n"
        "- **Each lane is independent.** Every beam and channel is processed "
        "separately.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CF32 | Beams or antennas x channels x samples x 2 "
        "polarizations. Axis roles, when set, must match these positions. The buffer "
        "must be contiguous. |\n"
        "| **Output** (out) | F32 | The input shape with samples divided by Integration"
        " Rate and 1 or 4 products in place of the polarizations. Sets `sampleAxis` to "
        "2 and `channelAxis` to 1, divides `sampleRate` by Integration Rate, and keeps "
        "other attributes. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Integration Rate** | 1 sample | 1 or more, dividing the samples | "
        "Consecutive samples summed into each output sample. |\n"
        "| **Output Polarizations** | Full Products | Total Power, Full Products | "
        "Total Power gives one product, XX plus YY. Full Products gives XX, YY, and the"
        " real and imaginary parts of XY. |\n"
        "| **Block Size** | 512 threads | 1 to 1024 | Threads per kernel block on CUDA."
        " Ignored on CPU. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume dual-polarization voltages from a Beamformer.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Full products | Defaults | 2 beams x 192 channels x 8,192 samples | 2 beams "
        "x 192 channels x 8,192 samples x 4 products |\n"
        "| Integrated total power | Total Power, Integration Rate 16 | 2 beams x 192 "
        "channels x 8,192 samples | 2 beams x 192 channels x 512 samples x 1 product |\n"
        "| Integrated full products | Integration Rate 64 | 3 beams x 192 channels x "
        "8,192 samples | 3 beams x 192 channels x 128 samples x 4 products |\n"
        "| One spectrum per buffer | Total Power, Integration Rate 8192 | 2 beams x 192"
        " channels x 8,192 samples | 2 beams x 192 channels x 1 sample x 1 product |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the time dimension is not divisible by the "
        "integration rate | The sample count is not a multiple of Integration Rate | "
        "Pick a divisor of the sample count, such as 16 for 8,192 samples. |\n"
        "| The block reports an unsupported data type, expecting CF32 | The input is "
        "CI8, such as straight from an ATA Receiver | Convert it with a Cast block, or "
        "detect after a Beamformer. |\n"
        "| The incoherent beam reads as power squared | The Beamformer stores power in "
        "that beam, and detection squares it again | Turn on Incoherent Beam Sqrt in "
        "the Beamformer. |\n"
        "| The block reports that the input needs 4 dimensions or 2 polarizations | The"
        " input is not shaped as antennas x channels x samples x 2 polarizations | Feed"
        " it dual-polarization voltages from a Beamformer or an ATA Receiver. |\n"
        "| The block reports that it expects a contiguous tensor | The input is "
        "strided, such as a Slice with Contiguous off | Turn on Contiguous in the "
        "Slice. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each sample yields the power of both polarizations and their cross product, "
        "and each group of consecutive samples is summed into one output sample. "
        "Nothing carries over between buffers. The model below covers one buffer in "
        "both modes.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def detect(x, integration_rate=1, output_polarizations=4):\n"
        "    a, f, t, _ = x.shape\n"
        "    x = x.reshape(a, f, t // integration_rate, integration_rate, 2)\n"
        "    xx = (np.abs(x[..., 0]) ** 2).sum(axis=-1)\n"
        "    yy = (np.abs(x[..., 1]) ** 2).sum(axis=-1)\n"
        "    if output_polarizations == 1:\n"
        "        return (xx + yy)[..., None].astype(np.float32)\n"
        "    xy = (x[..., 0] * np.conj(x[..., 1])).sum(axis=-1)\n"
        "    return np.stack([xx, yy, xy.real, xy.imag], axis=-1).astype(np.float32)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Detector guide](https://stelline.space/docs/block-detector) covers the "
        "products and tensor layouts.\n"
        "- **Beamformer** forms the beams it detects.\n"
        "- Filterbank files come from the **FBH5 Writer**.\n"
        "- **Integrator** sums along any axis, including across buffers."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_DETECTOR_BLOCK_HH
