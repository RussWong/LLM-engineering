#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/linear.h"
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/kernels/fused_transpose_and_remv_pad.h"
#include "src/kernels/concat_past_kv.h"
#include "src/kernels/repeat_kv.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_utils.h"
#include "src/models/llama/llama_params.h"
template<typename T>
class LLaMAContextAttentionLayer {
private:
    // this params are shared across all LLMs
    const int head_num;
    const int head_size;
    const int hidden_units;
    const int q_head_per_kv; //for GQA and MQA
    const int kv_head_num;
    float scale;
    // this params are only saw in llama and are unchanged 
    LLaMAAttentionStaticParams attn_static_params;
    cudaStream_t stream;
    BaseAllocator* allocator;
    // for linear and batchgemm
    cublasWrapper* cublas_wrapper;

    TensorWrapper<T>*  qkv_buf_wo_pad = nullptr;      
    TensorWrapper<T>*  q_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_buf_w_pad = nullptr;
    TensorWrapper<T>*  v_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_cache_buf = nullptr;
    TensorWrapper<T>*  v_cache_buf = nullptr;
    TensorWrapper<T>*  qk_buf = nullptr;
    TensorWrapper<T>*  qkv_buf_w_pad = nullptr;
    TensorWrapper<T>*  qkv_buf_wo_pad_1 = nullptr;      

public:
    LLaMAContextAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator);
    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }
    
    void allocForForward(LLaMAAttentionDynParams& params);
    void freeBuf();
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params);
    // whats the diff across these 3 max len:
    // max_seq_len is the max kv len considering context, ep. multiple epochs chat
    // max_q_len is the current max q len after padding in this batch
    // all kv cache is max seq len to save all kv cache in all epochs, but in context attention, all kv cache should be broadcast to adapt q as kv cache buf whose shape is max k len
    // so max k len is the max context len in cur batch  
    // void flashAttn();
};
