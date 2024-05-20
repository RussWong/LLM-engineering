#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"
#include "src/kernels/act_kernel.h"
#include "src/utils/macro.h"
template<typename T>
class LLaMAFFNLayer {
private:
    // this params are shared across all LLMs
    const int head_num;
    const int head_size;
    const int inter_size;
    const int hidden_units;
    int count = -1; // used to record layer index currently

    cudaStream_t stream;
    BaseAllocator* allocator;
    // for linear proj
    cublasWrapper* cublas_wrapper;

    // buffer
    // [2, num tokens, intersize]
    TensorWrapper<T>*  SwiGLU_input = nullptr;  //gate proj and up proj output buf   
    // [num tokens, intersize] 
    TensorWrapper<T>*  down_proj_input = nullptr;   


public:
    LLaMAFFNLayer(int head_num,
                    int head_size,
                    int inter_size,
                    cudaStream_t stream,
                    cublasWrapper* cublas_wrapper,
                    BaseAllocator* allocator);

    void allocForForward(LLaMAAttentionDynParams& params);
    void allocForForward(int batch_size);
    void freeBuf();
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAFFNWeights<T>& weights, LLaMAAttentionDynParams& params);
};
