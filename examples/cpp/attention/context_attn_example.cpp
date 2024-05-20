#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/attention/context_attention.h"

int main(int argc, char** argv)
{
    int head_num = 4;
    int kv_head_num = 2;
    int head_size = 8;
    int num_layers = 1;
    int max_seq_len = 12; // max context length for kv cache
    int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    int q_hidden_units = head_num * head_size;
    LLaMAAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope
    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 2;
    attn_dyn_params.num_tokens = 14;
    attn_dyn_params.max_q_len = 8;
    attn_dyn_params.max_k_len = 8; // max actual context length for cur batch
    bool is_free_buffer_after_fwd = true;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;
    // prepare input、weight and output data
    float* h_attention_input = (float*) malloc(sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens);
    float* d_attention_input;
    cudaMalloc((void**)&d_attention_input, sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens);
    for(int i = 0; i < q_hidden_units * attn_dyn_params.num_tokens; i++) { 
       h_attention_input[i] = 1.0f;
    }
    float* h_qkv_weights = (float*) malloc(sizeof(float) * q_hidden_units * hidden_units);
    float* d_qkv_weights;
    cudaMalloc((void**)&d_qkv_weights, sizeof(float) * q_hidden_units * hidden_units);
    for(int i = 0; i < hidden_units * q_hidden_units; i++) { 
       h_qkv_weights[i] = 1.0f;
    }
    float* h_mask = (float*) malloc(sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    float* d_mask;
    cudaMalloc((void**)&d_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    for(int i = 0; i < attn_dyn_params.max_q_len * attn_dyn_params.max_k_len * attn_dyn_params.batch_size; i++){
        h_mask[i] = 1.0f;
    }

    float* h_qkv_bias = (float*) malloc(sizeof(float) * hidden_units);
    float* d_qkv_bias;
    cudaMalloc((void**)&d_qkv_bias, sizeof(float) * hidden_units);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < hidden_units; i++){
        h_qkv_bias[i] = 2.0f;
    }
    //max_seq_len is the max kv cache len
    float* h_all_k_cache = (float*) malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float* d_all_k_cache;
    cudaMalloc((void**)&d_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    float* h_all_v_cache = (float*) malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float* d_all_v_cache;
    cudaMalloc((void**)&d_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size; i++) {
       h_all_k_cache[i] = 1.0f;
       h_all_v_cache[i] = 1.0f;
    }
    // padding to max_q_len
    int* h_padding_offset = (int*) malloc(sizeof(int) * attn_dyn_params.num_tokens);
    int* d_padding_offset;
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * attn_dyn_params.num_tokens);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < attn_dyn_params.num_tokens; i++) { // 3
       h_padding_offset[i] = i < 7 ? 0 : 1;// two seqlens are both 7, tokens num=14
    }
    int* h_history_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_history_len;
    cudaMalloc((void**)&d_history_len, sizeof(int) * attn_dyn_params.batch_size);
    int* h_input_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_input_len;
    cudaMalloc((void**)&d_input_len, sizeof(int) * attn_dyn_params.batch_size);
    int h_layer_id = 0;
    // int* d_layer_id;
    // cudaMalloc((void**)&d_layer_id, sizeof(int) * attn_dyn_params.batch_size);
    // note: cur_query_len and input_len are the same, I think
    // int* hcur_query_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    // int* dcur_query_len;
    // cudaMalloc((void**)&dcur_query_len, sizeof(int) * attn_dyn_params.batch_size);
    int* h_ctx_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_ctx_len;
    cudaMalloc((void**)&d_ctx_len, sizeof(int) * attn_dyn_params.batch_size);
    for(int i = 0; i < attn_dyn_params.batch_size; i++){
        h_history_len[i] = 0; // for kv cache cumsum seqlen and rope's timestep compute
        // h_layer_id[i] = 0;
        //hcur_query_len[i] = 7;
        h_input_len[i] = 7; // corresponding to padding offset
        h_ctx_len[i] = h_history_len[i] + h_input_len[i];
    }
    float* d_attention_output;
    cudaMalloc((void**)&d_attention_output, sizeof(float) * attn_dyn_params.num_tokens * q_hidden_units);
    float* d_output_weights;
    cudaMalloc((void**)&d_output_weights, sizeof(float) * q_hidden_units * q_hidden_units);

    // h2d
    cudaMemcpy(d_attention_input, h_attention_input, sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_weights, h_qkv_weights, sizeof(float) * q_hidden_units * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_history_len, h_history_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_layer_id, h_layer_id, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_len, h_input_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len, cudaMemcpyHostToDevice);
    // prepare input、weight and output tensor by std::initializer_list
    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    DataType type_int = getTensorType<int>();
    TensorWrapper<float>* attention_input = new TensorWrapper<float>(GPU, 
                                                                    type, 
                                                                    {attn_dyn_params.num_tokens, q_hidden_units}, 
                                                                    d_attention_input);
    TensorWrapper<float>* qkv_bias = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {hidden_units}, 
                                                              d_qkv_bias);
    TensorWrapper<int>* padding_offset = new TensorWrapper<int>(GPU, 
                                                              type_int, 
                                                              {attn_dyn_params.num_tokens}, 
                                                              d_padding_offset);
    TensorWrapper<int>* history_length = new TensorWrapper<int>(GPU, 
                                                              type_int, 
                                                              {attn_dyn_params.batch_size}, 
                                                              d_history_len);
    TensorWrapper<int>* input_length = new TensorWrapper<int>(GPU, 
                                                              type_int, 
                                                              {attn_dyn_params.batch_size}, 
                                                              d_input_len);
    TensorWrapper<int>* layer_id = new TensorWrapper<int>(CPU, 
                                                              type_int, 
                                                              {1}, 
                                                              &h_layer_id);
    TensorWrapper<int>* context_length = new TensorWrapper<int>(GPU, 
                                                              type_int, 
                                                              {attn_dyn_params.batch_size}, 
                                                              d_ctx_len);
    TensorWrapper<float>* attention_mask = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {attn_dyn_params.batch_size, attn_dyn_params.max_q_len, attn_dyn_params.max_k_len}, 
                                                              d_mask);
    TensorWrapper<float>* attention_output = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {attn_dyn_params.num_tokens, q_hidden_units}, 
                                                              d_attention_output);
    TensorWrapper<float>* all_k_cache = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                              d_all_k_cache);
    TensorWrapper<float>* all_v_cache = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                              d_all_v_cache);
    LLM_CHECK_WITH_INFO(attention_input->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(qkv_bias->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(padding_offset->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(layer_id->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "tensor inserted in tensormap is nullptr data!");
    LLM_CHECK_WITH_INFO(attention_mask->data != nullptr, "tensor inserted in tensormap is nullptr data!");

    TensorMap ctx_attn_inputs{
        {"attention_input", attention_input},
        {"qkv_bias", qkv_bias},
        {"padding_offset",padding_offset},
        {"history_length", history_length},
        {"input_length", input_length},
        {"layer_id", layer_id},
        {"context_length", context_length},
        {"attention_mask", attention_mask}
    };
    TensorMap ctx_attn_outputs{
        {"attention_output", attention_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };
    // weights are initialized in its constructor, see cpp/models/bert//bertlayerweight.cc
    LLaMAattentionWeights<float> ctx_attn_weights;
    WeightType wtype = getWeightType<float>();
    ctx_attn_weights.qkv.data = d_qkv_weights;
    ctx_attn_weights.qkv.shape = {q_hidden_units, hidden_units};
    ctx_attn_weights.qkv.type = wtype;
    ctx_attn_weights.qkv.bias = d_qkv_bias;
    ctx_attn_weights.output.data = d_output_weights;
    ctx_attn_weights.output.shape = {q_hidden_units, q_hidden_units};
    ctx_attn_weights.output.type = wtype;
    // init ctxAttn
    LLaMAContextAttentionLayer<float>* ctxAttn = new LLaMAContextAttentionLayer<float>(head_num,
                                                                                       kv_head_num,
                                                                                       head_size,
                                                                                       attn_static_params,
                                                                                       stream,
                                                                                       cublas_wrapper,
                                                                                       allocator);
//                                                                                       is_free_buffer_after_fwd);
    // forward
    ctxAttn->forward(ctx_attn_inputs, ctx_attn_outputs, ctx_attn_weights, attn_dyn_params, attn_static_params);
    // free buffer
    cudaDeviceSynchronize();
    free(h_attention_input);
    cudaFree(d_attention_input);
    free(h_qkv_bias);
    cudaFree(d_qkv_bias);
    free(h_all_k_cache);
    cudaFree(d_all_k_cache);
    free(h_all_v_cache);
    cudaFree(d_all_v_cache);
    free(h_padding_offset);
    cudaFree(d_padding_offset);
    free(h_history_len);
    cudaFree(d_history_len);
    free(h_input_len);
    cudaFree(d_input_len);
    // free(h_layer_id);
    // cudaFree(d_layer_id);
    free(h_ctx_len);
    cudaFree(d_ctx_len);
    cudaFree(d_attention_output);
    return 0;
}
