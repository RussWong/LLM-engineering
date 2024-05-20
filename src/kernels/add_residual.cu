#include <stdio.h>
#include "src/kernels/add_residual.h"
#include "src/utils/cuda_debug_utils.cuh"

// (RussWong)note: this kernel is used at the end of FFN in every decoder layer
template <typename T>
__global__ void AddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    T *residual,
    T *decoder_out, // [num tokens, hidden_units]
    int num_tokens,
    int hidden_units)
{
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x)
    {
        dout[i].x += rsd[i].x;
        dout[i].y += rsd[i].y;
        dout[i].z += rsd[i].z;
        dout[i].w += rsd[i].w;
    } // addresidual
}

template <>
__global__ void AddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    half *residual,
    half *decoder_out, // [num tokens, hidden_units]
    int num_tokens,
    int hidden_units)
{
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *dout = reinterpret_cast<Vec_t *>(decoder_out + batch_id * hidden_units);
    Vec_t *rsd = reinterpret_cast<Vec_t *>(residual + batch_id * hidden_units);
    for (int i = tid; i < hidden_units / vec_size; i += blockDim.x)
    {
        dout[i] = __hadd2(dout[i], rsd[i]);
    } // addresidual
}

template <typename T>
void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, 256 threads travrse hiddenunits eles recursely
    TensorWrapper<T> *residual,
    TensorWrapper<T> *decoder_out, // [num tokens, hidden_units]
    bool is_print
)
{
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<T>::size;
    dim3 grid(batch_size);
    dim3 block(256);
    AddResidual<T><<<grid, block>>>(residual->data,
                                    decoder_out->data,
                                    batch_size,
                                    hidden_units);
#ifdef PRINT_DATA
    if (is_print){
        print_data<<<1, 1>>>(decoder_out->data);
    }
#else
#endif
}
template void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    TensorWrapper<float> *residual,
    TensorWrapper<float> *decoder_out, // [num tokens, hidden_units]
    bool is_print
    );
template void launchAddResidual( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
    TensorWrapper<half> *residual,
    TensorWrapper<half> *decoder_out, // [num tokens, hidden_units]
    bool is_print
    );
