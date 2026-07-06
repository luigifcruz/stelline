#include "kernel.hh"

#include <cuda/std/complex>

// TODO: Make it faster.

namespace Jetstream::Modules {

namespace {

using U64 = std::uint64_t;

template<typename DstType>
__device__ DstType convertSample(const cuda::std::complex<int8_t>& value);

template<>
__device__ cuda::std::complex<int8_t> convertSample(const cuda::std::complex<int8_t>& value) {
    return value;
}

template<>
__device__ cuda::std::complex<float> convertSample(const cuda::std::complex<int8_t>& value) {
    return cuda::std::complex<float>(static_cast<float>(value.real()), static_cast<float>(value.imag()));
}

template<typename DstType>
__global__ void AtaGatherKernel(void* output,
                                void** input,
                                U64 numberOfPackets,
                                AtaReceiverBlockGeometry totalShape,
                                AtaReceiverBlockGeometry partialShape,
                                AtaReceiverBlockGeometry slotShape) {
    const U64 idx = static_cast<U64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= numberOfPackets) {
        return;
    }

    const U64 tChannels = totalShape.numberOfChannels;
    const U64 tSamples = totalShape.numberOfSamples;
    const U64 tPolarizations = totalShape.numberOfPolarizations;

    const U64 pAntennas = partialShape.numberOfAntennas;
    const U64 pChannels = partialShape.numberOfChannels;
    const U64 pSamples = partialShape.numberOfSamples;
    const U64 pPolarizations = partialShape.numberOfPolarizations;

    const U64 sAntennas = slotShape.numberOfAntennas;
    const U64 sChannels = slotShape.numberOfChannels;
    const U64 sSamples = slotShape.numberOfSamples;
    const U64 sPolarizations = slotShape.numberOfPolarizations;

    const U64 fragmentPolarizationIndex = idx % sPolarizations;
    const U64 fragmentSampleIndex = (idx / sPolarizations) % sSamples;
    const U64 fragmentChannelIndex = (idx / (sPolarizations * sSamples)) % sChannels;
    const U64 fragmentAntennaIndex = (idx / (sPolarizations * sSamples * sChannels)) % sAntennas;

    U64 baseOffset = 0;
    baseOffset += (fragmentAntennaIndex * pAntennas) * tChannels * tSamples * tPolarizations;
    baseOffset += (fragmentChannelIndex * pChannels) * tSamples * tPolarizations;
    baseOffset += (fragmentSampleIndex * pSamples) * tPolarizations;
    baseOffset += fragmentPolarizationIndex * pPolarizations;

    const auto* src = reinterpret_cast<const cuda::std::complex<int8_t>*>(input[idx]);
    auto* dst = reinterpret_cast<DstType*>(output) + baseOffset;

    for (U64 antenna = 0; antenna < pAntennas; antenna++) {
        for (U64 channel = 0; channel < pChannels; channel++) {
            for (U64 sample = 0; sample < pSamples; sample++) {
                for (U64 polarization = 0; polarization < pPolarizations; polarization++) {
                    const U64 srcOffset = antenna * pChannels * pSamples * pPolarizations +
                                          channel * pSamples * pPolarizations +
                                          sample * pPolarizations +
                                          polarization;
                    const U64 dstOffset = antenna * tChannels * tSamples * tPolarizations +
                                          channel * tSamples * tPolarizations +
                                          sample * tPolarizations +
                                          polarization;

                    dst[dstOffset] = convertSample<DstType>(src[srcOffset]);
                }
            }
        }
    }
}

}  // namespace

cudaError_t LaunchAtaGatherKernel(void* output,
                                  void** input,
                                  U64 numberOfPackets,
                                  const AtaReceiverBlockGeometry& totalShape,
                                  const AtaReceiverBlockGeometry& partialShape,
                                  const AtaReceiverBlockGeometry& slotShape,
                                  const std::string& outputType,
                                  cudaStream_t stream) {
    const dim3 threadsPerBlock(256, 1, 1);
    const dim3 blocksPerGrid((numberOfPackets + threadsPerBlock.x - 1) / threadsPerBlock.x, 1, 1);

    if (outputType == "CI8") {
        AtaGatherKernel<cuda::std::complex<int8_t>><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            output,
            input,
            numberOfPackets,
            totalShape,
            partialShape,
            slotShape);
        return cudaGetLastError();
    }

    if (outputType == "CF32") {
        AtaGatherKernel<cuda::std::complex<float>><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            output,
            input,
            numberOfPackets,
            totalShape,
            partialShape,
            slotShape);
        return cudaGetLastError();
    }

    return cudaErrorNotSupported;
}

}  // namespace Jetstream::Modules
