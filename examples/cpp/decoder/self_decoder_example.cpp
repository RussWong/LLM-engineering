#include <iostream>
#include <vector>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/decoder/self_decoder.h"
#include "src/utils/macro.h"

// current example dont consider layer_id in masked self attn
// now consider it
int main(){
    int h_step = 3;
    int head_num = 4;
    int kv_head_num = 2;
    int head_size = 8;
    int inter_size = 12;
    int num_layers = 32;
    int max_seq_len = 12;
    int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    int q_hidden_units = head_num * head_size;
    float rmsnorm_eps = 1e-6;
    LLaMAAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope
    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 2;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;  

    // prepare input、weight and output data
    float* h_decoder_input = (float*) malloc(sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);
    float* d_decoder_input;
    cudaMalloc((void**)&d_decoder_input, sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);
    for(int i = 0; i < q_hidden_units * attn_dyn_params.batch_size; i++) { 
       h_decoder_input[i] = rand() % 100 / (float)1000;
    }

    float* d_decoder_output;
    cudaMalloc((void**)&d_decoder_output, sizeof(float) * q_hidden_units * attn_dyn_params.batch_size);

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
    int layer_id = 0;
    bool* h_finished = (bool*) malloc(sizeof(bool) * attn_dyn_params.batch_size);
    bool* d_finished;
    cudaMalloc((void**)&d_finished, sizeof(bool) * attn_dyn_params.batch_size);
    for(int i = 0; i < attn_dyn_params.batch_size; i++){
        h_finished[i] = static_cast<bool>(0);
    }

    // weight
    // this weight is belong to llamaweight
    float* h_output_norm_weight = (float*)malloc(sizeof(float) * q_hidden_units);
    float* d_output_norm_weight;
    cudaMalloc((void**)&d_output_norm_weight, sizeof(float) * q_hidden_units);
    for(int i = 0; i < q_hidden_units; i++){
        h_output_norm_weight[i] = 2.0f;
    }

    float* h_attn_norm_weight = (float*)malloc(sizeof(float) * q_hidden_units);
    float* d_attn_norm_weight;
    cudaMalloc((void**)&d_attn_norm_weight, sizeof(float) * q_hidden_units);
    for(int i = 0; i < q_hidden_units; i++){
        h_attn_norm_weight[i] = 1.0f;
    }

    float* h_ffn_norm_weight = (float*)malloc(sizeof(float) * q_hidden_units);
    float* d_ffn_norm_weight;
    cudaMalloc((void**)&d_ffn_norm_weight, sizeof(float) * q_hidden_units);
    for(int i = 0; i < q_hidden_units; i++){
        h_ffn_norm_weight[i] = 1.0f;
    }

    float* h_qkv_weights = (float*) malloc(sizeof(float) * hidden_units * q_hidden_units);
    float* d_qkv_weights;
    cudaMalloc((void**)&d_qkv_weights, sizeof(float) * hidden_units * q_hidden_units);
    for(int i = 0; i < hidden_units * q_hidden_units; i++) { 
       h_qkv_weights[i] = 1.0f;
    }

    float* h_qkv_bias = (float*) malloc(sizeof(float) * hidden_units);
    float* d_qkv_bias;
    cudaMalloc((void**)&d_qkv_bias, sizeof(float) * hidden_units);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < hidden_units; i++){
        h_qkv_bias[i] = 2.0f;
    }

    float* h_output_weights = (float*) malloc(sizeof(float) * q_hidden_units * q_hidden_units);
    float* d_output_weights;
    cudaMalloc((void**)&d_output_weights, sizeof(float) * q_hidden_units * q_hidden_units);
    for(int i = 0; i < q_hidden_units * q_hidden_units; i++) { 
       h_output_weights[i] = 1.0f;
    }

    float* h_out_bias = (float*) malloc(sizeof(float) * head_num* head_size);
    float* d_out_bias;
    cudaMalloc((void**)&d_out_bias, sizeof(float) * head_num * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < head_num * head_size; i++){
        h_out_bias[i] = 2.0f;
    }
    float* d_ffn_gate, *d_ffn_up, *d_ffn_down, *d_ffn_down_bias;
    float* h_ffn_gate_up = (float*) malloc(sizeof(float) * hidden_units * 2 * inter_size);
    // float* h_ffn_up = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* h_ffn_down = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* h_ffn_down_bias = (float*) malloc(sizeof(float) * hidden_units);
    cudaMalloc((void**)&d_ffn_gate, sizeof(float) * hidden_units * 2 * inter_size);
    // cudaMalloc((void**)&d_ffn_up, sizeof(float) * hidden_units * inter_size);
    cudaMalloc((void**)&d_ffn_down, sizeof(float) * hidden_units * inter_size);
    cudaMalloc((void**)&d_ffn_down_bias, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units * 2 * inter_size; i++){
        h_ffn_gate_up[i] = 2.0f;
    }
    for(int i = 0; i < hidden_units * inter_size; i++){
        h_ffn_down[i] = 2.0f;
        if (i < hidden_units){
            h_ffn_down_bias[i] = 0.0f;
        }
    }  

    // h2d
    cudaMemcpy(d_decoder_input, h_decoder_input, sizeof(float) * q_hidden_units * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);

    cudaMemcpy(d_finished, h_finished, sizeof(bool) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_norm_weight, h_output_norm_weight, sizeof(float) * q_hidden_units, cudaMemcpyHostToDevice);

    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    DataType type_int = getTensorType<int>();
    DataType type_bool = getTensorType<bool>();
    std::vector<LlamaLayerWeight<float>*> layerWeights;
    WeightType wtype = getWeightType<float>();
    layerWeights.reserve(num_layers);
    for(int i = 0; i < num_layers; i++) {
        layerWeights[i] = new LlamaLayerWeight<float>(head_num, kv_head_num,
                                               head_size, inter_size, wtype,
                                               /*attn_bias*/true);
        layerWeights[i]->loadWeights();
    }
    TensorWrapper<float>* decoder_input = new TensorWrapper<float>(GPU, 
                                                                    type, 
                                                                    {attn_dyn_params.batch_size, q_hidden_units}, 
                                                                    d_decoder_input);
    TensorWrapper<int>* step = new TensorWrapper<int>(CPU, 
                                                        type_int, 
                                                        {1}, 
                                                        &h_step);
    TensorWrapper<bool>* finished = new TensorWrapper<bool>(GPU, 
                                                            type_bool, 
                                                            {attn_dyn_params.batch_size}, 
                                                            d_finished);
    TensorWrapper<int>* layer = new TensorWrapper<int>(CPU, 
                                                            type_int, 
                                                            {1}, 
                                                            &layer_id);
    TensorWrapper<float>* output_norm_weight = new TensorWrapper<float>(GPU, 
                                                            type, 
                                                            {q_hidden_units}, 
                                                            d_output_norm_weight);
    TensorWrapper<float>* decoder_output = new TensorWrapper<float>(GPU, 
                                                                    type, 
                                                                    {attn_dyn_params.batch_size, q_hidden_units}, 
                                                                    d_decoder_output);
    TensorWrapper<float>* key_cache = new TensorWrapper<float>(GPU, 
                                                                type, 
                                                                {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                                d_all_k_cache);
    TensorWrapper<float>* value_cache = new TensorWrapper<float>(GPU, 
                                                                type, 
                                                                {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                                d_all_v_cache);
    LLM_CHECK_WITH_INFO(decoder_input->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");


    TensorMap decoder_inputs{
        {"decoder_input",decoder_input},
        // {"sequence_lengths", Tensor(GPU, type, {hidden_units}, )},
        // {"total_padding_len", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, )},
        {"step",step},// a batch shared same step, dim=1 tensor can locate on CPU, no need GPU
        {"finished",finished},
        {"layer_id",layer},
        {"output_norm_weight",output_norm_weight}//located at llamaweights class, rather not llamalayerweigths
    };
    TensorMap decoder_outputs{
        {"decoder_output",decoder_output},
        {"all_k_cache",key_cache},
        {"all_v_cache",value_cache}
    };

    LlamaSelfDecoder<float>* selfDecoder = new LlamaSelfDecoder<float>(head_num,
                                                            kv_head_num,
                                                            head_size,
                                                            inter_size,
                                                            num_layers,
                                                            attn_static_params,
                                                            rmsnorm_eps,
                                                            stream,
                                                            cublas_wrapper,
                                                            allocator);
    selfDecoder->forward(decoder_inputs, layerWeights, decoder_outputs, attn_dyn_params);
    cudaDeviceSynchronize();
    free(h_decoder_input);
    free(h_all_k_cache);
    free(h_all_v_cache);
    free(h_finished);
    free(h_qkv_weights);
    free(h_output_weights);
    free(h_qkv_bias);
    cudaFree(d_decoder_input);
    cudaFree(d_all_k_cache);
    cudaFree(d_all_v_cache);
    cudaFree(d_finished);
    cudaFree(d_qkv_weights);
    cudaFree(d_output_weights);
    cudaFree(d_qkv_bias);
  
}
