#include "kernel.hh"

// TODO: Improve performance.

namespace stelline::operators::transport {

__global__ void Kernel(void* output, void** input, uint64_t numberOfPackets,
                       BlockShape totalShape, BlockShape partialShape, BlockShape slots) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= numberOfPackets) {
        return;
    }

    // Breakout the shapes.

    int tF = totalShape.numberOfChannels;
    int tT = totalShape.numberOfSamples;
    int tP = totalShape.numberOfPolarizations;

    int pA = partialShape.numberOfAntennas;
    int pF = partialShape.numberOfChannels;
    int pT = partialShape.numberOfSamples;
    int pP = partialShape.numberOfPolarizations;

    int fA = slots.numberOfAntennas;
    int fF = slots.numberOfChannels;
    int fT = slots.numberOfSamples;
    int fP = slots.numberOfPolarizations;

    // Transform the fragment index into shape.

    int fragmentPolarizationIndex = idx % fP;
    int fragmentTimeIndex = (idx / fP) % fT;
    int fragmentFrequencyIndex = (idx / (fP * fT)) % fF;
    int fragmentAntennaIndex = (idx / (fP * fT * fF)) % fA;

    // Transform the shape into the defragmented data offset.

    int defragmentationOffset = 0;
    defragmentationOffset += (fragmentAntennaIndex * pA) * tF * tT * tP;
    defragmentationOffset += (fragmentFrequencyIndex * pF) * tT * tP;
    defragmentationOffset += (fragmentTimeIndex * pT) * tP;
    defragmentationOffset += (fragmentPolarizationIndex * pP);

    // Get the source and destination pointers.

    uint16_t* src = ((uint16_t**)input)[idx];
    uint16_t* dst = &((uint16_t*)output)[defragmentationOffset];

    // Copy the fragment into the defragmented data.

    for (int A = 0; A < pA; A++) {
        for (int F = 0; F < pF; F++) {
            for (int T = 0; T < pT; T++) {
                for (int P = 0; P < pP; P++) {
                    int srcOffset = A * pF * pT * pP + F * pT * pP + T * pP + P;
                    int dstOffset = A * tF * tT * tP + F * tT * tP + T * tP + P;

                    dst[dstOffset] = src[srcOffset];
                }
            }
        }
    }
}

cudaError_t LaunchKernel(void* output, void** input, uint64_t numberOfPackets,
                         BlockShape totalShape, BlockShape partialShape, BlockShape slots,
                         cudaStream_t stream) {
    dim3 threadsPerBlock = {512, 1, 1};
    dim3 blocksPerGrid = {
        (static_cast<int32_t>(numberOfPackets) + threadsPerBlock.x - 1) / threadsPerBlock.x,
        1,
        1
    };

    Kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(output, input, numberOfPackets,
                                                          totalShape, partialShape, slots);

    return cudaGetLastError();
}

}  // namespace stelline::operators::transport