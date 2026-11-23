#ifndef STELLINE_INTEGRATOR_BLOCK_HH
#define STELLINE_INTEGRATOR_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Integrator : public Block::Config {
    U64 size = 1;  // Number of indices to integrate within one block.
    U64 rate = 1;  // Number of blocks to integrate together.
    U64 axis = 2;  // The block axis to integrate on, defaulting to T
    U64 blockSize = 512;

    JST_BLOCK_TYPE(stelline_integrator);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(size, rate, axis, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Integrator",
        "Integrates (sums) samples along an axis.",
        "# Integrator\n"
        "\n"
        "Sums groups of consecutive values along one axis, and can add several buffers "
        "into one output. It usually follows a Detector or a Beamformer, to raise the "
        "signal-to-noise ratio and lower the data rate.\n"
        "\n"
        "- **Rate holds the output back.** Blocks downstream run once every Rate "
        "buffers, on the summed result.\n"
        "- **Timed windows start on a grid.** Windows of Rate timed buffers open only "
        "at a `timestamp` that is a multiple of their total `duration`.\n"
        "- **A gap restarts the window.** A jump in `timestamp` drops the partial sum, "
        "and repeated or older buffers are skipped.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32, CF32, or CI8 | Any shape with more than Axis "
        "dimensions. The buffer must be contiguous. |\n"
        "| **Output** (out) | CF32 | The input shape with the Axis length divided by "
        "Size, and every axis role kept. When Axis is the `sampleAxis`, `sampleRate` is"
        " divided by Size. The `timestamp` and `duration` cover the whole window. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Size** | 1 | 1 or more, dividing the Axis length | Consecutive values "
        "summed into one along Axis. |\n"
        "| **Rate** | 1 | 1 or more | Buffers added into each output. |\n"
        "| **Axis** | 2 | Below the input rank | Axis to sum along. The default is the "
        "samples axis of a four-axis voltage buffer. |\n"
        "| **Block Size** | 512 threads | 1 to 1,024 threads | Threads per CUDA block. "
        "Ignored on CPU. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Coarser time bins | Size 16 | 1 antenna x 192 channels x 8,192 samples x 2 "
        "polarizations | 1 antenna x 192 channels x 512 samples x 2 polarizations |\n"
        "| Sum buffers over time | Rate 8 | 1 antenna x 192 channels x 512 samples x 4 "
        "polarizations | The same shape, once every 8 buffers |\n"
        "| Both at once | Size 8, Rate 4 | 2 antennas x 192 channels x 1,024 samples x "
        "2 polarizations | 2 antennas x 192 channels x 128 samples x 2 polarizations, "
        "once every 4 buffers |\n"
        "| Collapse the band | Axis 1, Size 192 | 1 antenna x 192 channels x 1,024 "
        "samples x 2 polarizations | 1 antenna x 1 channel x 1,024 samples x 2 "
        "polarizations |\n"
        "| One stream | Axis 0, Size 4 | 8,192 real samples at 2 MS/s | 2,048 complex "
        "samples at 500 kS/s |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The block reports that the selected axis must exist on the input | Axis is "
        "at or above the input rank, such as the default 2 on a one-dimensional stream "
        "| Set Axis below the rank, such as 0. |\n"
        "| The block reports that the axis must be a whole multiple of the integration "
        "size | The Axis length is not a multiple of Size | Pick a Size that divides "
        "the length. |\n"
        "| Blocks downstream never run | The input carries only one of `timestamp` and "
        "`duration`, or a zero or non-U64 value | Set both attributes upstream as U64, "
        "or neither. |\n"
        "| The level grows with Size and Rate | The block sums, it does not average | "
        "Scale by 1 / (Size x Rate) with a Multiply Constant block. |\n"
        "| The block reports an unsupported data type | The input is not F32, CF32, or "
        "CI8 | Convert it with a Cast block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each output value sums consecutive input values along the axis after a "
        "conversion to complex, and later buffers add into the same output. Timed "
        "buffers group into windows aligned to their total duration, so a gap or a "
        "stale buffer never mixes into a window. The model below covers untimed "
        "buffers, with the running sum passed in and returned.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def integrate(x, acc=None, count=0, size=1, rate=1, axis=2):\n"
        "    n = x.shape[axis]\n"
        "    shape = x.shape[:axis] + (n // size, size) + x.shape[axis + 1:]\n"
        "    y = x.astype(np.complex64).reshape(shape).sum(axis=axis + 1)\n"
        "    acc = y if count == 0 else acc + y\n"
        "    count = (count + 1) % rate\n"
        "    return (acc if count == 0 else None), acc, count\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Integrator guide](https://stelline.space/docs/block-integrator) covers "
        "timing windows in detail.\n"
        "- **Stacker** joins buffers along an axis instead of summing.\n"
        "- **Detector** sums power over time before this block.\n"
        "- **Multiply Constant** turns the sum into an average."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_INTEGRATOR_BLOCK_HH
