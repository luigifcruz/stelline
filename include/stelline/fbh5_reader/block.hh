#ifndef STELLINE_FBH5_READER_BLOCK_HH
#define STELLINE_FBH5_READER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Fbh5Reader : public Block::Config {
    std::string filepath = "";
    U64 batchSize = 256;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(fbh5_reader);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "FBH5 Reader",
        "Source that reads beamformer filterbank data from an FBH5 file.",
        "# FBH5 Reader\n"
        "Reads channelized filterbank data from an FBH5 (HDF5) archive and emits it as "
        "F32 spectrogram chunks. FBH5 files store data in the dataset `/data` with shape "
        "[ntimes, nifs, nchans].\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the `.fbh5` or `.h5` file.\n"
        "- **Batch Size**: Number of time rows to read per processing cycle.\n"
        "- **Loop**: Restart from the beginning when the end of file is reached.\n"
        "- **Playing**: Pause or resume reading.\n\n"

        "## Outputs\n"
        "- **Signal**: F32 tensor [batchSize, nifs x nchans] — one row per time sample.\n\n"

        "## Useful For\n"
        "- Offline / archival playback of beamformed spectra into a flowgraph.\n"
        "- Driving FRBNN inference from recorded FBH5 files.\n\n"

        "## Implementation\n"
        "Reads with the POSIX HDF5 VFD (no GPUDirect Storage required), so it runs on any "
        "system that has libhdf5 — including non-NVIDIA hosts."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_FBH5_READER_BLOCK_HH
