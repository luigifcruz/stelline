#ifndef STELLINE_UVH5_READER_BLOCK_HH
#define STELLINE_UVH5_READER_BLOCK_HH

#include <string>

#include <jetstream/block.hh>

namespace Jetstream::Blocks {

struct Uvh5Reader : public Block::Config {
    std::string filepath = "";
    U64 batchSize = 256;
    bool loop = true;
    bool playing = true;

    JST_BLOCK_TYPE(uvh5_reader);
    JST_BLOCK_DOMAIN("Stelline");
    JST_BLOCK_PARAMS(filepath, batchSize, loop, playing);
    JST_BLOCK_DESCRIPTION(
        "UVH5 Reader",
        "Source that reads correlator visibilities from a UVH5 file.",
        "# UVH5 Reader\n"
        "Reads visibility data from a UVH5 (pyuvdata / HDF5) archive and emits it as CF32 "
        "chunks. UVH5 files store visibilities in `/Data/visdata` with shape "
        "[Nblts, Nspws, Nfreqs, Npols] complex64.\n\n"

        "## Arguments\n"
        "- **File Path**: Path to the `.uvh5` or `.h5` file.\n"
        "- **Batch Size**: Number of baseline-time (blt) entries to read per cycle.\n"
        "- **Loop**: Restart from the beginning when the end of file is reached.\n"
        "- **Playing**: Pause or resume reading.\n\n"

        "## Outputs\n"
        "- **Signal**: CF32 tensor [batchSize, Nspws x Nfreqs, Npols] of visibilities.\n\n"

        "## Useful For\n"
        "- Offline / archival playback of correlator visibilities into a flowgraph.\n\n"

        "## Implementation\n"
        "Reads with the POSIX HDF5 VFD (no GPUDirect Storage required), so it runs on any "
        "system that has libhdf5 — including non-NVIDIA hosts."
    );
};

}  // namespace Jetstream::Blocks

#endif  // STELLINE_UVH5_READER_BLOCK_HH
