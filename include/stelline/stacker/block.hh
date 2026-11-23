#ifndef STELLINE_STACKER_BLOCK_HH
#define STELLINE_STACKER_BLOCK_HH

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Stacker : public Block::Config {
    U64 axis = 0;
    U64 ratio = 1;
    U64 copySizeThreshold = 512;
    U64 blockSize = 512;

    JST_BLOCK_TYPE(stelline_stacker);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(axis, ratio, copySizeThreshold, blockSize);
    JST_BLOCK_DESCRIPTION(
        "Stacker",
        "Stacks the input along a specified axis.",
        "# Stacker\n"
        "\n"
        "Joins successive buffers end to end along one axis, so each output holds "
        "several buffers. It usually follows a Beamformer or a Detector, to build the "
        "longer blocks that a writer or a correlator expects.\n"
        "\n"
        "- **Ratio holds the output back.** Blocks downstream run once every Ratio "
        "buffers, on the joined result.\n"
        "- **The default Axis is the outermost.** Set Axis to 2 to join the samples of "
        "a four-axis voltage buffer.\n"
        "- **Timed windows start on a grid.** Windows of Ratio timed buffers open only "
        "at a `timestamp` that is a multiple of their total `duration`.\n"
        "- **A gap restarts the window.** A jump in `timestamp` drops the partial "
        "output, and repeated or older buffers are skipped.\n"
        "\n"
        "## Ports\n"
        "\n"
        "| Port | Type | Shape |\n"
        "|---|---|---|\n"
        "| **Input** (in) | F32, CF32, or CI8 | Any shape with more than Axis "
        "dimensions. The buffer must be contiguous. |\n"
        "| **Output** (out) | Same as the input | The input shape with the Axis length "
        "x Ratio, and every axis role kept. The `timestamp` and `duration` cover the "
        "whole window, and other attributes are kept. |\n"
        "\n"
        "## Settings\n"
        "\n"
        "| Setting | Default | Range | Effect |\n"
        "|---|---|---|---|\n"
        "| **Axis** | 0 | Below the input rank | Axis the buffers are joined along, in "
        "arrival order. |\n"
        "| **Ratio** | 1 | 1 or more | Buffers joined into each output. |\n"
        "| **Copy Threshold** | 512 | 0 or more | Row width in values below which a "
        "kernel copies each buffer, instead of a strided copy. The row is the Axis and "
        "every later axis. Ignored on CPU. |\n"
        "| **Block Size** | 512 threads | 1 to 1,024 threads | Threads per CUDA block "
        "when a kernel copies. Ignored on CPU. |\n"
        "\n"
        "## Recipes\n"
        "\n"
        "| Goal | Settings | Input | Output |\n"
        "|---|---|---|---|\n"
        "| Longer voltage blocks | Axis 2, Ratio 8 | 2 antennas x 192 channels x 1,024 "
        "samples x 2 polarizations | 2 antennas x 192 channels x 8,192 samples x 2 "
        "polarizations, once every 8 buffers |\n"
        "| History of spectra | Axis 2, Ratio 64 | 1 antenna x 192 channels x 1 sample "
        "x 4 polarizations | 1 antenna x 192 channels x 64 samples x 4 polarizations |\n"
        "| Visibilities over time | Axis 2, Ratio 10 | 406 baselines x 192 channels x 1"
        " sample x 4 products | 406 baselines x 192 channels x 10 samples x 4 products "
        "|\n"
        "| Longer stream | Ratio 4 | 2,048 samples | 8,192 samples, once every 4 "
        "buffers |\n"
        "\n"
        "## Troubleshooting\n"
        "\n"
        "| Symptom | Cause | Fix |\n"
        "|---|---|---|\n"
        "| The output grows along antennas instead of samples | Axis is 0, the default "
        "| Set Axis to 2 for a four-axis voltage buffer. |\n"
        "| The block reports that the selected axis must exist on the input | Axis is "
        "at or above the input rank | Set Axis below the rank. |\n"
        "| Blocks downstream never run | The input carries only one of `timestamp` and "
        "`duration`, or a zero or non-U64 value | Set both attributes upstream as U64, "
        "or neither. |\n"
        "| The block reports that the axis ratio must be greater than 0 | Ratio is 0 | "
        "Set Ratio to 1 or more. |\n"
        "| The block reports an unsupported data type | The input is not F32, CF32, or "
        "CI8 | Convert it with a Cast block. |\n"
        "\n"
        "## Under the Hood\n"
        "\n"
        "Each buffer is copied into its own slot along the axis, so the output holds "
        "the buffers in arrival order. Timed buffers group into windows aligned to "
        "their total duration, so a gap or a stale buffer never mixes into a window. "
        "The model below covers untimed buffers, with the collected buffers passed in "
        "and returned.\n"
        "\n"
        "```python\n"
        "import numpy as np\n"
        "\n"
        "def stack(x, parts=(), axis=0, ratio=1):\n"
        "    parts = list(parts) + [x]\n"
        "    if len(parts) < ratio:\n"
        "        return None, parts\n"
        "    return np.concatenate(parts, axis=axis), []\n"
        "```\n"
        "\n"
        "## See Also\n"
        "\n"
        "- The [Stacker guide](https://stelline.space/docs/block-stacker) covers timing"
        " windows in detail.\n"
        "- **Integrator** sums buffers instead of joining them.\n"
        "- **Reshape** splits the joined axis into new dimensions."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_STACKER_BLOCK_HH
