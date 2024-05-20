#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/attention/masked_self_attention.h"
// current example dont consider layer_id in masked self attn
int main(){
    int h_step = 3;
    int head_num = 4;
    int kv_head_num = 2;
    int head_size = 8;
    int num_layers = 1;
    int max_seq_len = 12;
    int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    int q_hidden_units = head_num * head_size;
    LLaMAAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope
    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 2;
    // attn_dyn_params.num_tokens = 14;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;
    // prepare input、weight and output data
    float* h_attention_input = (float*) malloc(sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);
    float* d_attention_input;
    cudaMalloc((void**)&d_attention_input, sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);
    for(int i = 0; i < q_hidden_units * attn_dyn_params.batch_size; i++) { 
       h_attention_input[i] = 1.0f;
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
    int h_layer_id = 0;
    bool* h_finished = (bool*) malloc(sizeof(bool) * attn_dyn_params.batch_size);
    bool* d_finished;
    cudaMalloc((void**)&d_finished, sizeof(bool) * attn_dyn_params.batch_size);
    for(int i = 0; i < attn_dyn_params.batch_size; i++){
        h_finished[i] = static_cast<bool>(0);
    }

    float* h_qkv_weights = (float*) malloc(sizeof(float) * q_hidden_units * hidden_units);
    float* d_qkv_weights;
    cudaMalloc((void**)&d_qkv_weights, sizeof(float) * q_hidden_units * hidden_units);
    for(int i = 0; i < hidden_units * q_hidden_units; i++) { 
       h_qkv_weights[i] = 1.0f;
    }    

    float* h_output_weights = (float*) malloc(sizeof(float) * q_hidden_units * q_hidden_units);
    float* d_output_weights;
    cudaMalloc((void**)&d_output_weights, sizeof(float) * q_hidden_units * q_hidden_units);
    for(int i = 0; i < q_hidden_units * q_hidden_units; i++) { 
       h_output_weights[i] = 1.0f;
    }

    float* h_qkv_bias = (float*) malloc(sizeof(float) * hidden_units);
    float* d_qkv_bias;
    cudaMalloc((void**)&d_qkv_bias, sizeof(float) * hidden_units);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < hidden_units; i++){
        h_qkv_bias[i] = 2.0f;
    }

    float* d_attention_output;
    cudaMalloc((void**)&d_attention_output, sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);

    CHECK(cudaMemcpy(d_attention_input, h_attention_input, sizeof(float) * q_hidden_units * attn_dyn_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_finished, h_finished, sizeof(bool) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_weights, h_qkv_weights, sizeof(float) * q_hidden_units * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_output_weights, h_output_weights, sizeof(float) * q_hidden_units * q_hidden_units, cudaMemcpyHostToDevice));

    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();
    LLaMAattentionWeights<float> self_attn_weights;
    WeightType wtype = getWeightType<float>();
    self_attn_weights.qkv.data = d_qkv_weights;
    self_attn_weights.qkv.shape = {q_hidden_units, hidden_units};
    self_attn_weights.qkv.type = wtype;
    self_attn_weights.qkv.bias = d_qkv_bias;
    self_attn_weights.output.data = d_output_weights;
    self_attn_weights.output.shape = {q_hidden_units, q_hidden_units};
    self_attn_weights.output.type = wtype;
    TensorWrapper<float>* attention_input = new TensorWrapper<float>(GPU, 
                                                                    type, 
                                                                    {attn_dyn_params.batch_size, q_hidden_units}, 
                                                                    d_attention_input);
    TensorWrapper<int>* step = new TensorWrapper<int>(CPU, 
                                                        type_int, 
                                                        {1}, 
                                                        &h_step);
    TensorWrapper<bool>* finished = new TensorWrapper<bool>(GPU, 
                                                            type_bool, 
                                                            {attn_dyn_params.batch_size}, 
                                                            d_finished);
    TensorWrapper<int>* layer_id = new TensorWrapper<int>(CPU, 
                                                            type_int, 
                                                            {1}, 
                                                            &h_layer_id);
    TensorWrapper<float>* attention_output = new TensorWrapper<float>(GPU, 
                                                                    type, 
                                                                    {attn_dyn_params.batch_size, q_hidden_units}, 
                                                                    d_attention_output);
    TensorWrapper<float>* key_cache = new TensorWrapper<float>(GPU, 
                                                                type, 
                                                                {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                                d_all_k_cache);
    TensorWrapper<float>* value_cache = new TensorWrapper<float>(GPU, 
                                                                type, 
                                                                {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                                d_all_v_cache);
    LLM_CHECK_WITH_INFO(attention_input->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer_id->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap masked_attn_inputs{
        {"attention_input", attention_input},
        // {"sequence_lengths", Tensor(GPU, type, {hidden_units}, d_qkv_bias)},
        // {"total_padding_len", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_padding_offset)},
        {"step", step},// a batch shared same step, dim=1 tensor can locate on CPU, no need GPU
        {"finished", finished},
        {"layer_id", layer_id},
    };
    TensorMap masked_attn_outputs{
        {"attention_output", attention_output},
        {"all_k_cache", key_cache},
        {"all_v_cache", value_cache}
    };

    LLaMASelfAttentionLayer<float>* self_attn_layer = new LLaMASelfAttentionLayer<float>( head_num,
                                                                            kv_head_num,
                                                                            head_size,
                                                                            attn_static_params,
                                                                            stream,
                                                                            cublas_wrapper,
                                                                            allocator);
    self_attn_layer->forward(masked_attn_inputs,
                             masked_attn_outputs,
                             self_attn_weights,
                             attn_dyn_params);
    cudaDeviceSynchronize();
    free(h_attention_input);
    free(h_all_k_cache);
    free(h_all_v_cache);
    free(h_finished);
    free(h_qkv_weights);
    free(h_output_weights);
    free(h_qkv_bias);
    cudaFree(d_attention_input);
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_finished);
    cudaFree(d_qkv_weights);
    cudaFree(d_output_weights);
    cudaFree(d_qkv_bias);
}
