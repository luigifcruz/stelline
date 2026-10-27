#include "kernel.hh"

#include <cuda/std/complex>

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
__global__ void AtaScatterKernel(const void* const* inputs,
                                 void* const* outputs,
                                 U64 numberOfPackets,
                                 AtaReceiverBlockGeometry totalShape,
                                 AtaReceiverBlockGeometry partialShape) {
    const U64 idx = blockIdx.x;
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

    const auto* src = reinterpret_cast<const cuda::std::complex<int8_t>*>(inputs[idx]);
    auto* dst = reinterpret_cast<DstType*>(outputs[idx]);

    const U64 inputChannelStride = pSamples * pPolarizations;
    const U64 inputAntennaStride = pChannels * inputChannelStride;
    const U64 outputChannelStride = tSamples * tPolarizations;
    const U64 outputAntennaStride = tChannels * outputChannelStride;

    for (U64 antenna = 0; antenna < pAntennas; antenna++) {
        for (U64 element = threadIdx.x; element < inputAntennaStride; element += blockDim.x) {
            const U64 channel = element / inputChannelStride;
            const U64 withinChannel = element % inputChannelStride;
            U64 outputWithinChannel = withinChannel;
            if (pPolarizations != tPolarizations) {
                outputWithinChannel = (withinChannel / pPolarizations) * tPolarizations +
                                      withinChannel % pPolarizations;
            }

            const U64 srcOffset = antenna * inputAntennaStride + element;
            const U64 dstOffset = antenna * outputAntennaStride +
                                  channel * outputChannelStride + outputWithinChannel;
            dst[dstOffset] = convertSample<DstType>(src[srcOffset]);
        }
    }
}

}  // namespace

cudaError_t LaunchAtaScatterKernel(const void* const* inputs,
                                   void* const* outputs,
                                   U64 numberOfPackets,
                                   const AtaReceiverBlockGeometry& totalShape,
                                   const AtaReceiverBlockGeometry& partialShape,
                                   const std::string& outputType,
                                   cudaStream_t stream) {
    const dim3 threadsPerBlock(256, 1, 1);
    const dim3 blocksPerGrid(numberOfPackets, 1, 1);

    if (outputType == "CI8") {
        AtaScatterKernel<cuda::std::complex<int8_t>><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            inputs,
            outputs,
            numberOfPackets,
            totalShape,
            partialShape);
        return cudaGetLastError();
    }

    if (outputType == "CF32") {
        AtaScatterKernel<cuda::std::complex<float>><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            inputs,
            outputs,
            numberOfPackets,
            totalShape,
            partialShape);
        return cudaGetLastError();
    }

    return cudaErrorNotSupported;
}

}  // namespace Jetstream::Modules
