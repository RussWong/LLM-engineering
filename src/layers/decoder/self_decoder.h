#pragma once
#include "src/kernels/fused_decoder_self_attention.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/kernels/rmsnorm_kernel.h"
#include "src/kernels/add_residual.h"
#include "src/layers/attention/masked_self_attention.h"
#include "src/layers/ffn/ffn.h"
#include "src/weights/llama/llama_weights.h"
#include "src/utils/tensor.h"

// layer weights is ready at the model_utils.h                                                                                                                                           by loadweights in onellm.cpp, outside of the decoder
template <typename T>
class LlamaSelfDecoder
{
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int inter_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps;

    cudaStream_t stream;
    cublasWrapper *cublas_wrapper;
    BaseAllocator *allocator;

    TensorWrapper<T> *decoder_residual;

    LLaMASelfAttentionLayer<T> *selfAttn;
    LLaMAFFNLayer<T> *ffn;
    DataType data_type;

public:
    LlamaSelfDecoder(int head_num,
                     int kv_head_num,
                     int head_size,
                     int inter_size,
                     int num_layer,
                     const LLaMAAttentionStaticParams &attn_params,
                     float rmsnorm_eps,
                     cudaStream_t stream,
                     cublasWrapper *cublas_wrapper,
                     BaseAllocator *allocator) : head_num(head_num),
                                                 head_size(head_size),
                                                 inter_size(inter_size),
                                                 hidden_units(head_num * head_size),
                                                 num_layer(num_layer),
                                                 rmsnorm_eps(rmsnorm_eps),
                                                 data_type(getTensorType<float>()),
                                                 stream(stream),
                                                 cublas_wrapper(cublas_wrapper),
                                                 allocator(allocator)
    {
        selfAttn = new LLaMASelfAttentionLayer<T>(head_num,
                                                  kv_head_num,
                                                  head_size,
                                                  attn_params,
                                                  stream,
                                                  cublas_wrapper,
                                                  allocator);

        ffn = new LLaMAFFNLayer<T>(head_num,
                                   head_size,
                                   inter_size,
                                   stream,
                                   cublas_wrapper,
                                   allocator);
    };
    void allocForForward(LLaMAAttentionDynParams &dyn_params);
    void freeBuf();
    void forward(TensorMap &input_tensors, const std::vector<LlamaLayerWeight<T> *> &layerWeights, TensorMap &output_tensors, LLaMAAttentionDynParams &dyn_params);
};
