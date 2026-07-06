#ifndef STELLINE_UVH5_WRITER_BLOCK_HH
#define STELLINE_UVH5_WRITER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Uvh5Writer : public Block::Config {
    std::string filepath = "./file.uvh5";
    U64 dspChannelizationRate = 1;
    U64 dspIntegrationRate = 1;
    bool overwrite = false;
    bool recording = false;

    JST_BLOCK_TYPE(uvh5_writer);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, dspChannelizationRate, dspIntegrationRate, overwrite, recording);
    JST_BLOCK_DESCRIPTION(
        "UVH5 Writer",
        "Sink that writes pre-arranged correlation tensors to UVH5.",
        "# UVH5 Writer\n"
        "The UVH5 Writer block writes pre-arranged visibility tensors to a UVH5 file using "
        "GPUDirect Storage. Input tensors must be shaped as [time, baselines, frequencies, "
        "polarizations] with one time sample and four polarization products per write. "
        "Observatory and observation metadata are read from the flowgraph environment, "
        "typically populated by the Nexus Bridge block.\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the output UVH5 file.\n"
        "- **DSP Channelization Rate**: Upstream channelization factor applied to each coarse channel.\n"
        "- **DSP Integration Rate**: Upstream integration factor used to derive the integration timespan.\n"
        "- **Overwrite**: Whether to overwrite the file if it already exists.\n"
        "- **Recording**: Start or stop recording to the file.\n\n"

        "## Useful For\n"
        "- Recording correlated visibilities for radio interferometry imaging.\n"
        "- Producing UVH5 files compatible with pyuvdata tooling.\n"
        "- Streaming GPU-resident correlation products straight to disk.\n\n"

        "## Examples\n"
        "- Record visibilities to UVH5:\n"
        "  Config: File Path='observation.uvh5', Overwrite=true, Recording=true\n"
        "  Input: CF32[1, 406, 192, 4] -> Appended to the file each cycle.\n\n"

        "## Implementation\n"
        "Input Buffer -> UVH5 Module -> HDF5 GDS File\n"
        "1. Loads observatory, antenna, and band metadata from the flowgraph environment.\n"
        "2. Builds the UVH5 header and opens the file with the GPUDirect Storage HDF5 driver.\n"
        "3. Appends each input tensor to the visibility dataset directly from GPU memory.\n"
        "4. Refreshes pointing and IERS metadata using the input timestamp attribute."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_UVH5_WRITER_BLOCK_HH
