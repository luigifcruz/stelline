#include "modifiers.hh"

template<typename T, int32_t RANK>
cudaError_t Permutation(DLManagedTensor* dst, const DLManagedTensor* src) {
    auto t1 = matx::tensor_t<T, RANK>{};
    auto t2 = matx::tensor_t<T, RANK>{};

    matx::make_tensor(t1, *src);
    matx::make_tensor(t2, *dst);

    // TODO: Switch to einsum if performance is necessary.
    //(t2 = matx::cutensor::einsum(einsum_str, t1)).run();
    (t2 = matx::permute(t1, {2, 0, 1, 3})).run();

    return cudaSuccess;
}

namespace stelline::operators::filesystem {

cudaError_t BlockPermutation(DLManagedTensor* dst, const DLManagedTensor* src) {
    if (src->dl_tensor.ndim != 4) {
        return cudaErrorInvalidValue;
    }

    switch (src->dl_tensor.dtype.code) {
        case 2:  //kDLFloat:
            return Permutation<float, 4>(dst, src);
        case 5:  //kDLComplex:
            return Permutation<cuda::std::complex<float>, 4>(dst, src);
        default:
            return cudaErrorInvalidValue;
    }
}

}  // namespace stelline::operators::filesystem
