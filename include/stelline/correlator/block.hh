#ifndef STELLINE_CORRELATOR_BLOCK_HH
#define STELLINE_CORRELATOR_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Correlator : public Block::Config {
    U64 integrationRate = 1;
    U64 conjugateAntennaIndex = 1;
    std::string calculationMode = "double_precision_fp";

    JST_BLOCK_TYPE(stelline_correlator);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_NODE_SIZE(M);
    JST_BLOCK_PARAMS(integrationRate, conjugateAntennaIndex, calculationMode);
    JST_BLOCK_DESCRIPTION(
        "Correlator",
        "Correlates voltages into baseline visibilities.",
        "# Correlator\n"
        "\n"
        "Cross-correlates every pair of antennas, each antenna with itself included, "
        "into visibilities summed over the samples of each buffer. It usually follows "
        "an ATA Receiver and feeds a UVH5 Writer.\n"
        "\n"
        "- **Integration Rate holds the output back.** Blocks downstream run once every"
        " Integration Rate buffers, on the summed visibilities.\n"
        "- **Timed windows start on a grid.** Windows of several timed buffers open "
        "only at a `timestamp` that is a multiple of their total `duration`.\n"
        "- **A gap restarts the window.** A jump in `timestamp` drops the partial sum, "
        "and repeated or older buffers are skipped.\n"
        "- **Integer input is not scaled.** Values from a CI8 input correlate as whole "
        "numbers, unlike in the Beamformer.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | CI8 or CF32 | Antennas x channels x samples x 2 "
        "polarizations. Axis roles, when set, must match these positions. The buffer "
        "must be contiguous. |\n"
        "| **Output** (out) | CF32 | Baselines x channels x 1 sample x 4 products (XX, "
        "XY, YX, YY), with antennas x (antennas + 1) / 2 baselines. Divides "
        "`sampleRate` by samples x Integration Rate. The `timestamp` and `duration` "
        "cover the whole window. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Integration Rate** | 1 buffer | 1 or more | Input buffers summed into each"
        " output. |\n"
        "| **Conjugate Antenna** | Antenna B | Antenna A, Antenna B | Antenna of each "
        "pair that is conjugated. Antenna A flips the sign of every imaginary part. |\n"
        "| **Calculation Mode** | Double Precision FP | Integer, Single Precision FP, "
        "Double Precision FP | Precision of the sum over samples. Integer needs CI8 "
        "input and at most 65,535 samples. On CUDA, CI8 input with a multiple of 4 "
        "samples up to 65,535 always sums as integers. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "All rows assume 192 channels x 2 polarizations from an ATA Receiver.\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Visibilities per buffer | Defaults | 28 antennas x 8,192 samples | 406 "
        "baselines x 192 channels x 1 sample x 4 products, every buffer |\n"
        "| Integrate eight buffers | Integration Rate 8 | 28 antennas x 8,192 samples |"
        " 406 baselines x 192 channels x 1 sample x 4 products, every eighth buffer |\n"
        "| Exact integer sums | Integer | 28 antennas x 8,192 samples, CI8 | 406 "
        "baselines x 192 channels x 1 sample x 4 products |\n"
        "| Float input | Single Precision FP | 20 antennas x 8,192 samples, CF32 | 210 "
        "baselines x 192 channels x 1 sample x 4 products |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that integer calculation is not supported for CF32 input |"
        " Calculation Mode is Integer and the input is CF32 | Pick Single Precision FP "
        "or Double Precision FP. |\n"
        "| The block reports that Integer mode overflows above 65535 time samples | "
        "Calculation Mode is Integer and a CI8 buffer holds more samples | Pick a "
        "floating-point mode, or shorten the buffer upstream. |\n"
        "| Blocks downstream never run | The input has only one of `timestamp` and "
        "`duration`, or a zero or non-integer value, so every buffer is skipped | Fix "
        "the timing attributes upstream, or remove both. |\n"
        "| The block reports that the input needs 4 dimensions or 2 polarizations | The"
        " input is not shaped as antennas x channels x samples x 2 polarizations | Feed"
        " it dual-polarization voltages from an ATA Receiver. |\n"
        "| The visibility phases have the opposite sign from a reference | The "
        "reference conjugates the other antenna of each pair | Switch Conjugate "
        "Antenna. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each antenna pair multiplies one antenna by the conjugate of the other and "
        "sums the four polarization products over every sample of the buffer. "
        "Integration adds the sums of consecutive buffers into one output, aligned to "
        "the timing grid when buffers carry timestamps. The model below covers one "
        "buffer, without integration or the calculation precision.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def correlate(x, conjugate_antenna=\"B\"):\n"
        "    if x.dtype == np.int8:\n"
        "        x = x[..., 0] + 1j * x[..., 1]\n"
        "    a, b = np.triu_indices(x.shape[0])\n"
        "    v = np.einsum(\"lftp,lftq->lfpq\", x[a], np.conj(x[b]))\n"
        "    v = v.reshape(len(a), x.shape[1], 1, 4)\n"
        "    return (v if conjugate_antenna == \"B\" else np.conj(v)).astype(np.complex64)\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Correlator guide](https://stelline.space/docs/block-correlator) covers "
        "the calculation modes and timing windows.\n"
        "- Visibility files come from the **UVH5 Writer**.\n"
        "- **Beamformer** sums antennas into beams instead.\n"
        "- **Integrator** sums detected power across buffers."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_CORRELATOR_BLOCK_HH
