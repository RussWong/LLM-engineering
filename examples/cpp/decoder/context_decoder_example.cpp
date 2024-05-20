#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <random>
#include "src/layers/decoder/context_decoder.h"
#include "src/utils/macro.h"
#include "src/models/tokenizer.h"
#include "src/kernels/input_embedding.h"
#include "src/weights/llama/embedding_weights.h"

int main(int argc, char** argv)
{
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;

    int head_num = 32;
    int kv_head_num = 32;
    int head_size = 128;
    int inter_size = 11008;
    int num_layers = 32;
    int max_seq_len = 64;
    int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    int q_hidden_units = head_num * head_size;
    float rmsnorm_eps = 1e-6;
    LLaMAAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope

    std::string input = "how old are you";
    Tokenizer tokenizer;
    tokenizer.Initialize("/home/llama2-7b-tokenizer.bin");
    std::vector<int> res = tokenizer.Encode(input);
    std::cout << "input ids length is " << res.size() << "\n";
    int *h_input_ids_buf_;
    h_input_ids_buf_ =
        allocator->Malloc(h_input_ids_buf_, sizeof(int) * res.size(), true);
    for (int i = 0; i < res.size(); i++)
    {
        h_input_ids_buf_[i] = res[i]; // [max_context_token_nums_]
    }
    // ensure prepared all needed input buffer
    int index = 0;
    int ret;
    int context_length_ = res.size();
    int history_length_ = 0;
    int cur_input_length = res.size(); // res.size() is the input ids len, which is the real input len, rather not len of input string
    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 1;
    attn_dyn_params.num_tokens = cur_input_length;          
    attn_dyn_params.max_q_len = attn_dyn_params.num_tokens; // 指一个batch中的q的最大长度，因为此时不支持batch，所以就等于cur input len
    attn_dyn_params.max_k_len = context_length_;            //这个指max context len，指当前batch的动态最大上下文长度
    // retString为当前轮次对话的所有token string
    std::string retString = "";

    TensorWrapper<int>* input_ids = new TensorWrapper<int>(GPU, getTensorType<int>(), {cur_input_length});
    input_ids->data = allocator->Malloc(input_ids->data, sizeof(int) * cur_input_length, false);
    CHECK(cudaMemcpy(input_ids->data,                                    
                     h_input_ids_buf_,                                   
                     sizeof(int) * cur_input_length, 
                     cudaMemcpyHostToDevice));
    TensorWrapper<float>* decoder_input = new TensorWrapper<float>(GPU, getTensorType<float>(), {/*token num*/  attn_dyn_params.num_tokens, q_hidden_units});
    decoder_input->data = allocator->Malloc(decoder_input->data, sizeof(float) * attn_dyn_params.num_tokens * q_hidden_units, false); 
    float* embedding = (float*)malloc(sizeof(float) * 32000 * 4096);
    for(int i = 0; i < 32000 * 4096; i++){
        embedding[i] = rand() % 100 / (float)100000;
    }
    float* d_embedding;
    CHECK(cudaMalloc((void**)&d_embedding, sizeof(float) * 32000 * 4096));
    CHECK(cudaMemcpy(d_embedding, embedding, sizeof(float) * 32000 * 4096, cudaMemcpyHostToDevice));
    EmbeddingWeight<float> embed_table;
    WeightType wtype = getWeightType<float>();
    embed_table.shape = {32000, 4096};
    embed_table.type = wtype;
    embed_table.data = d_embedding;
    launchInputEmbedding(input_ids, decoder_input, &embed_table);

    // float* h_decoder_input = (float*) malloc(sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens);
    // float* d_decoder_input;
    // cudaMalloc((void**)&d_decoder_input, sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens);
    
    // for(int i = 0; i < q_hidden_units * attn_dyn_params.num_tokens; i++) { 
    //    h_decoder_input[i] = rand() % 100 / (float)(100000);
    // }
    
    float* d_decoder_output;
    cudaMalloc((void**)&d_decoder_output, sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens);

    float* h_mask = (float*) malloc(sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    float* d_mask;
    cudaMalloc((void**)&d_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    for(int i = 0; i < attn_dyn_params.max_q_len * attn_dyn_params.max_k_len * attn_dyn_params.batch_size; i++){
        h_mask[i] = 1.0f;
    }

    //max_seq_len is the max kv cache len
    float* h_all_k_cache = (float*) malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float* d_all_k_cache;
    cudaMalloc((void**)&d_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    float* h_all_v_cache = (float*) malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float* d_all_v_cache;
    cudaMalloc((void**)&d_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size; i++) {
       h_all_k_cache[i] = rand() % 100 / (float)100000;
       h_all_v_cache[i] = rand() % 100 / (float)100000;
    }
    // padding to max_q_len
    int* h_padding_offset = (int*) malloc(sizeof(int) * attn_dyn_params.num_tokens);
    int* d_padding_offset;
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * attn_dyn_params.num_tokens);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < attn_dyn_params.num_tokens; i++) { // 3
       //h_padding_offset[i] = i < 8 ? 0 : 1;// two seqlens are both 7, tokens num=14
        h_padding_offset[i] = 0;
    }
    int* h_history_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_history_len;
    cudaMalloc((void**)&d_history_len, sizeof(int) * attn_dyn_params.batch_size);
    int* h_input_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_input_len;
    cudaMalloc((void**)&d_input_len, sizeof(int) * attn_dyn_params.batch_size);
    int* h_ctx_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_ctx_len;
    cudaMalloc((void**)&d_ctx_len, sizeof(int) * attn_dyn_params.batch_size);
    for(int i = 0; i < attn_dyn_params.batch_size; i++){
        h_history_len[i] = 0; // for kv cache cumsum seqlen and rope's timestep compute
        h_input_len[i] = cur_input_length;
        h_ctx_len[i] = context_length_;
    }
    // weight
    // this weight is belong to llamaweight class
    float* h_output_norm_weight = (float*)malloc(sizeof(float) * q_hidden_units);
    float* d_output_norm_weight;
    cudaMalloc((void**)&d_output_norm_weight, sizeof(float) * q_hidden_units);
    for(int i = 0; i < q_hidden_units; i++){
        h_output_norm_weight[i] = rand() % 100 / (float)100000;
    }
 
    // h2d
    // cudaMemcpy(d_decoder_input, h_decoder_input, sizeof(float) * q_hidden_units * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_history_len, h_history_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_len, h_input_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len, cudaMemcpyHostToDevice);

    cudaMemcpy(d_output_norm_weight, h_output_norm_weight, sizeof(float) * q_hidden_units, cudaMemcpyHostToDevice);
    int layer_id = 0;
    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    DataType type_int = getTensorType<int>();
    std::vector<LlamaLayerWeight<float>*> layerWeights;
    layerWeights.reserve(num_layers);
    for(int i = 0; i < num_layers; i++) {
        layerWeights[i] = new LlamaLayerWeight<float>(head_num, kv_head_num,
                                               head_size, inter_size, wtype,
                                               /*attn_bias*/false);
        layerWeights[i]->loadWeights();
    }
    // TensorWrapper<float>* decoder_input = new TensorWrapper<float>(GPU, 
    //                                                                 type, 
    //                                                                 {attn_dyn_params.num_tokens, q_hidden_units}, 
    //                                                                 d_decoder_input);
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
    TensorWrapper<int>* layer = new TensorWrapper<int>(CPU, 
                                                              type_int, 
                                                              {1}, 
                                                              &layer_id);
    TensorWrapper<int>* context_length = new TensorWrapper<int>(GPU, 
                                                              type_int, 
                                                              {attn_dyn_params.batch_size}, 
                                                              d_ctx_len);
    TensorWrapper<float>* attention_mask = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {attn_dyn_params.batch_size, attn_dyn_params.max_q_len, attn_dyn_params.max_k_len}, 
                                                              d_mask);
    TensorWrapper<float>* decoder_output = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {attn_dyn_params.num_tokens, q_hidden_units}, 
                                                              d_decoder_output);
    TensorWrapper<float>* all_k_cache = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                              d_all_k_cache);
    TensorWrapper<float>* all_v_cache = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, 
                                                              d_all_v_cache);
    TensorWrapper<float>* output_norm_weight = new TensorWrapper<float>(GPU, 
                                                              type, 
                                                              {q_hidden_units}, 
                                                              d_output_norm_weight);
    LLM_CHECK_WITH_INFO(decoder_input->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
//    LLM_CHECK_WITH_INFO(padding_offset->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(history_length->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
//    LLM_CHECK_WITH_INFO(attention_mask->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(layer->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(context_length->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(output_norm_weight->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(input_length->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    std::cout << "in context decoder example cpp: " << layer->DeviceString() << "\n";    
    TensorMap decoder_inputs{
        {"decoder_input", decoder_input},
//        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", input_length},
        {"context_length", context_length},
 //       {"attention_mask", attention_mask},
        {"output_norm_weight", output_norm_weight},//located at llamaweights class, rather not llamalayerweigths
        {"layer_id", layer}
    };
    //output buffer and input buffer are shared to reuse buffer between layers
    //I dont rewrite Tensor's copy constructor, default shallow copy, that can share buffer, which is I want
    TensorMap decoder_outputs{
        {"decoder_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    LlamaContextDecoder<float>* ctxDecoder = new LlamaContextDecoder<float>(head_num,
                                                            kv_head_num,
                                                            head_size,
                                                            inter_size,
                                                            num_layers,
                                                            attn_static_params,
                                                            rmsnorm_eps,
                                                            stream,
                                                            cublas_wrapper,
                                                            allocator);
    ctxDecoder->forward(decoder_inputs, layerWeights, decoder_outputs, attn_dyn_params);
    cudaDeviceSynchronize();
    // gpu buffer can be released in corresponding class
    // free(h_decoder_input);
    // cudaFree(d_decoder_input);
    DeviceSyncAndCheckCudaError();
    free(h_all_k_cache);
    cudaFree(d_all_k_cache);
    DeviceSyncAndCheckCudaError();
    free(h_all_v_cache);
    cudaFree(d_all_v_cache);
    DeviceSyncAndCheckCudaError();
    free(h_padding_offset);
    cudaFree(d_padding_offset);
    DeviceSyncAndCheckCudaError();
    free(h_history_len);
    cudaFree(d_history_len);
    DeviceSyncAndCheckCudaError();
    free(h_ctx_len);
    cudaFree(d_ctx_len);
    DeviceSyncAndCheckCudaError();
    free(h_input_len);
    cudaFree(d_input_len);
    DeviceSyncAndCheckCudaError();
    free(h_mask);
    cudaFree(d_mask); 
    DeviceSyncAndCheckCudaError();
    free(h_output_norm_weight);
    cudaFree(d_output_norm_weight);
    DeviceSyncAndCheckCudaError();
    free(h_input_ids_buf_);
    free(embedding);
    cudaFree(d_embedding);
}
