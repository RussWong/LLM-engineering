#include <algorithm> // std::fill_n
#include <iostream>  // snprintf
#include <math.h>    // expf, log
#include <stdlib.h>  // rand
#include <string>    // std::string
#include <vector>    // std::vector

#include "src/kernels/fused_decoder_self_attention.h"
#include "src/utils/macro.h"

// bug1: MUST add CHECK to cudaMemcpy to see if its work well
// (RussWong)note: this CPU implementation still exists bugs.
// when you are implementing LLMs inference on CPU, you can reuse the CPU kernel and test its correctness
// `./test_fused_decoder_attention` to test fp32 kernel
template <typename T>
void CPUMaskedAttn(T *q,
                   T *k,
                   T *v,
                   T *k_cache,
                   T *v_cache,
                   float *mha_output,
                   const int batch_size,
                   const int num_heads,
                   const int head_size,
                   int step)
{
    int batch_stride = num_heads * head_size;
    int head_stride = head_size;
    int cache_offset = batch_size * batch_stride;
    int block_nums = batch_size * num_heads;
    float scale = rsqrt(float(head_size));

    const T *q_mem = q;
    const T *k_mem = k;
    const T *v_mem = v;

    // tmp buffer
    float *sqk = (float *)malloc(sizeof(float) * (block_nums * (3 * head_size + step)));
    float *sq = sqk;
    float *sk = sq + block_nums * head_size;
    float *logits = sk + block_nums * head_size;
    float *sv = logits + block_nums * step;
    for (int batch_id = 0; batch_id < batch_size; batch_id++)
    {
        for (int head_id = 0; head_id < num_heads; head_id++)
        {
            float row_max = 0.0f;
            for (int iter = 0; iter < step; iter++)
            {
                float attn_score = 0.0f;
                for (int tid = 0; tid < head_size; tid++)
                {
                    int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                    // note: sq and sk's offset should be qkv_offset , not tid
                    sk[qkv_offset] = (float)k_cache[iter * cache_offset + qkv_offset];
                    // when final step, update k cache
                    if (iter == step - 1)
                    {
                        // TODO: update k cache with k with bias add
                        k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
                        sk[qkv_offset] = (float)k_mem[qkv_offset];
                    }

                    sq[qkv_offset] = (float)q_mem[qkv_offset];
                    float qk = sq[qkv_offset] * sk[qkv_offset] * scale;
                    // block reduce using multi warp reduce
                    // TODO: maybe broadcast the attn score to each thread of the block in blockreducesum
                    attn_score += qk;
                }
                // note: logtis's offset should be as follow, not should mul head size with iter
                // debug info,printf("every step/seqlen attn score = %f\n", attn_score);
                logits[batch_id * num_heads * step + head_id * step + iter] = attn_score;
                // softmax(logits), logits.shape = [bs, num heads, 1, step]
                row_max = std::max(attn_score, row_max);
            }
            //            printf("all step/seqlen(one row) max attn score = %f\n", row_max);
            float fenzi = 0.0f;
            float fenmu = 0.0f;
            for (int iter = 0; iter < step; iter++)
            { // row
                fenzi = expf(logits[batch_id * num_heads * step + head_id * step + iter] - row_max);
                fenmu += fenzi;
            }
            for (int iter = 0; iter < step; iter++)
            { // row
                logits[batch_id * num_heads * step + head_id * step + iter] = (float)(fenzi / fenmu);
                //              printf("logits=%f\n", fenzi / fenmu);
            }
            // logits*V = [bs, num heads, 1, step] * [mx_seq_len or step, bs, num heads, head size]
            // for(int iter = 0; iter < step; iter++) {
            for (int tid = 0; tid < head_size; tid++)
            {
                float O = 0.0f;
                int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                for (int iter = 0; iter < step; iter++)
                {
                    sv[qkv_offset] = (float)v_cache[iter * cache_offset + qkv_offset];
                    // when final step, update k cache
                    if (iter == step - 1)
                    {
                        // TODO: update k cache with k with bias add
                        v_cache[iter * cache_offset + qkv_offset] = v_mem[qkv_offset];
                        sv[qkv_offset] = (float)v_mem[qkv_offset];
                    }
                    O += sv[qkv_offset] * logits[batch_id * num_heads * step + head_id * step + iter];
                    //                printf("logits[%d]=%f, sv[%d]=%f, O=%f\n", iter, logits[iter], qkv_offset, sv[qkv_offset], O);
                }
                mha_output[qkv_offset] = O;
            }
        }
    }

    free(sqk);
}
template <typename T>
bool CheckResult(float *CPUoutput, T *GPUoutput, int output_size)
{
    for (int i = 0; i < output_size; i++)
    {
        float GPUres = (float)GPUoutput[i];
        if (fabs(CPUoutput[i] - GPUres) > 1e-6)
        {
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUres);
            return false;
        }
    }
    return true;
}
// (RussWong)note: when we develop fp16 fused decoder attention, we can enable the disabled macro to generalize fp16 an fp32 UT
// #define LAUNCH_FUSED_ATTN(dtype)                                                                                                      \
//     dtype *h_qkv;                                                                                                                     \
//     dtype *d_qkv;                                                                                                                     \
//     int qkv_size = batch_size * (2 * kv_num_heads + num_heads) * head_size;                                                           \
//     h_qkv = (dtype *)malloc(sizeof(dtype) * qkv_size);                                                                                \
//     cudaMalloc((void **)&d_qkv, sizeof(dtype) * qkv_size);                                                                            \
//     dtype *h_kcache;                                                                                                                  \
//     dtype *d_kcache;                                                                                                                  \
//     int kcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                            \
//     h_kcache = (dtype *)malloc(sizeof(dtype) * kcache_size);                                                                          \
//     cudaMalloc((void **)&d_kcache, sizeof(dtype) * kcache_size);                                                                      \
//     dtype *h_vcache;                                                                                                                  \
//     dtype *d_vcache;                                                                                                                  \
//     int vcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                            \
//     h_vcache = (dtype *)malloc(sizeof(dtype) * vcache_size);                                                                          \
//     cudaMalloc((void **)&d_vcache, sizeof(dtype) * vcache_size);                                                                      \
//     for (int i = 0; i < qkv_size; i++)                                                                                                \
//     {                                                                                                                                 \
//         if (i < batch_size * num_heads * head_size)                                                                                   \
//         {                                                                                                                             \
//             if (i < batch_size * num_heads * head_size / 2)                                                                           \
//             {                                                                                                                         \
//                 h_qkv[i] = (dtype)(i + 1);                                                                                            \
//             }                                                                                                                         \
//             else                                                                                                                      \
//             {                                                                                                                         \
//                 h_qkv[i] = (dtype)(i + 1) / (dtype)10;                                                                                       \
//             }                                                                                                                         \
//         }                                                                                                                             \
//         else if (i < batch_size * (num_heads + kv_num_heads) * head_size)                                                             \
//         {                                                                                                                             \
//             if (i < batch_size * (num_heads + kv_num_heads / 2) * head_size)                                                          \
//             {                                                                                                                         \
//                 h_qkv[i] = (dtype)(i + 5);                                                                                            \
//             }                                                                                                                         \
//             else                                                                                                                      \
//             {                                                                                                                         \
//                 h_qkv[i] = (dtype)(i + 1) / (dtype)10;                                                                                       \
//             }                                                                                                                         \
//         }                                                                                                                             \
//         else if (i < batch_size * (num_heads + kv_num_heads * 2) * head_size)                                                         \
//         {                                                                                                                             \
//             if (i < batch_size * (num_heads + kv_num_heads + kv_num_heads / 2) * head_size)                                           \
//             {                                                                                                                         \
//                 h_qkv[i] = (dtype)(i - 3);                                                                                            \
//             }                                                                                                                         \
//             else                                                                                                                      \
//             {                                                                                                                         \
//                 h_qkv[i] = (dtype)(i - 7) / (dtype)10;                                                                                       \
//             }                                                                                                                         \
//         }                                                                                                                             \
//         printf("h_qkv[%d]= %f \n", i, (dtype)h_qkv[i]);                                                                                      \
//     }                                                                                                                                 \
//     dtype *h_q = h_qkv;                                                                                                               \
//     dtype *h_k = h_q + batch_size * num_heads * head_size;                                                                            \
//     dtype *h_v = h_k + batch_size * (kv_num_heads + num_heads) * head_size;                                                           \
//     for (int i = 0; i < (kcache_size * h_step) / max_seq_len; i++)                                                                    \
//     {                                                                                                                                 \
//         if (i < kcache_size / 2)                                                                                                      \
//         {                                                                                                                             \
//             h_kcache[i] = (dtype)(i + 1);                                                                                             \
//             h_vcache[i] = (dtype)(i + 1);                                                                                             \
//         }                                                                                                                             \
//         else                                                                                                                          \
//         {                                                                                                                             \
//             h_kcache[i] = (dtype)(i - kcache_size / 2 + 1) / (dtype)10;                                                                      \
//             h_vcache[i] = (dtype)(i - kcache_size / 2 + 1) / (dtype)10;                                                                      \
//         }                                                                                                                             \
//         printf("h_kcache[%d]= %f \n", i, (dtype)h_kcache[i]);                                                                                \
//         printf("h_vcache[%d]= %f \n", i, (dtype)h_vcache[i]);                                                                                \
//     }                                                                                                                                 \
//     dtype *h_o;                                                                                                                       \
//     dtype *d_o;                                                                                                                       \
//     int o_size = batch_size * num_heads * head_size;                                                                                  \
//     h_o = (dtype *)malloc(sizeof(dtype) * o_size);                                                                                    \
//     cudaMalloc((void **)&d_o, sizeof(dtype) * o_size);                                                                                \
//     bool *h_finished = (bool *)malloc(sizeof(bool) * batch_size);                                                                     \
//     bool *d_finished;                                                                                                                 \
//     cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size);                                                                      \
//     for (int i = 0; i < batch_size; i++)                                                                                              \
//     {                                                                                                                                 \
//         h_finished[i] = static_cast<bool>(0);                                                                                         \
//     }                                                                                                                                 \
//     dtype *h_qkv_bias = (dtype *)malloc(sizeof(dtype) * (2 * kv_num_heads + num_heads) * head_size);                                  \
//     dtype *d_qkv_bias;                                                                                                                \
//     cudaMalloc((void **)&d_qkv_bias, sizeof(dtype) * (2 * kv_num_heads + num_heads) * head_size);                                     \
//     for (int i = 0; i < (2 * kv_num_heads + num_heads) * head_size; i++)                                                              \
//     {                                                                                                                                 \
//         h_qkv_bias[i] = (dtype)0.0f;                                                                                                  \
//     }                                                                                                                                 \
//     cudaMemcpy(d_qkv, h_qkv, sizeof(dtype) * batch_size * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);        \
//     cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(dtype) * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);           \
//     cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice);                                            \
//     cudaMemcpy(d_kcache, h_kcache, sizeof(dtype) * kcache_size, cudaMemcpyHostToDevice);                                              \
//     cudaMemcpy(d_vcache, h_vcache, sizeof(dtype) * vcache_size, cudaMemcpyHostToDevice);                                              \
//     DataType type = getTensorType<dtype>();                                                                                           \
//     DataType type_bool = getTensorType<bool>();                                                                                       \
//     DataType type_int = getTensorType<int>();                                                                                         \
//     TensorWrapper<dtype> *qkv = new TensorWrapper<dtype>(GPU, type, {batch_size, num_heads + 2 * kv_num_heads, head_size}, d_qkv);    \
//     TensorWrapper<dtype> *kcache = new TensorWrapper<dtype>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_kcache); \
//     TensorWrapper<dtype> *vcache = new TensorWrapper<dtype>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_vcache); \
//     TensorWrapper<bool> *finished = new TensorWrapper<bool>(GPU, type_bool, {batch_size}, d_finished);                                \
//     TensorWrapper<int> *step = new TensorWrapper<int>(CPU, type_int, {1}, &h_step);                                                   \
//     TensorWrapper<int> *layer_id = new TensorWrapper<int>(CPU, type_int, {1}, &h_layer_id);                                           \
//     TensorWrapper<dtype> *mha_output = new TensorWrapper<dtype>(GPU, type, {batch_size, num_heads, head_size}, d_o);                  \
//     BaseWeight<dtype> qkv_weight;                                                                                                     \
//     qkv_weight.bias = d_qkv_bias;                                                                                                     \
//     LLaMAAttentionStaticParams params;                                                                                                \
//     params.rotary_embedding_dim = rotary_embedding_dim;                                                                               \
//     params.rotary_embedding_base = rotary_embedding_base;                                                                             \
//     params.max_position_embeddings = max_position_embeddings;                                                                         \
//     params.use_dynamic_ntk = false;                                                                                                   \
//     launchDecoderMaskedMHA(qkv, qkv_weight, layer_id, kcache, vcache, finished, step, mha_output, params);                            \
//     CHECK(cudaMemcpy(h_o, d_o, sizeof(dtype) * o_size, cudaMemcpyDeviceToHost));                                                      \
//     float *CPU_output = (float *)malloc(sizeof(float) * o_size);                                                                      \
//     CPUMaskedAttn<dtype>(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, head_size, h_step);                    \
//     bool is_true = CheckResult<dtype>(CPU_output, h_o, o_size);                                                                       \
//     if (is_true)                                                                                                                      \
//     {                                                                                                                                 \
//         printf("test passed");                                                                                                        \
//     }                                                                                                                                 \
//     else                                                                                                                              \
//     {                                                                                                                                 \
//         printf("test failed");                                                                                                        \
//     }                                                                                                                                 \
//     free(h_qkv);                                                                                                                      \
//     free(h_kcache);                                                                                                                   \
//     free(h_vcache);                                                                                                                   \
//     free(h_o);                                                                                                                        \
//     free(CPU_output);                                                                                                                 \
//     free(h_finished);                                                                                                                 \
//     cudaFree(d_finished);                                                                                                             \
//     cudaFree(d_qkv);                                                                                                                  \
//     cudaFree(d_o);                                                                                                                    \
//     cudaFree(d_kcache);                                                                                                               \
//     cudaFree(d_vcache);

int main(int argc, char *argv[])
{
    constexpr int batch_size = 1;
    constexpr int head_size = 4;
    constexpr int num_heads = 2;
    constexpr int kv_num_heads = 2;
    constexpr int max_seq_len = 4;
    int h_step = 4;
    int h_layer_id = 0;
    int rotary_embedding_dim = 128;
    float rotary_embedding_base = 10000;
    int max_position_embeddings = 2048;
    bool use_dynamic_ntk = false; // for dyn scaling rope
    float *h_qkv;                                                                                                                                 
    float *d_qkv;                                                                                                                                 
    int qkv_size = batch_size * (2 * kv_num_heads + num_heads) * head_size;                                                                       
    h_qkv = (float *)malloc(sizeof(float) * qkv_size);                                                                                            
    cudaMalloc((void **)&d_qkv, sizeof(float) * qkv_size);                                                                                        
    float *h_kcache;                                                                                                                              
    float *d_kcache;                                                                                                                              
    int kcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_kcache = (float *)malloc(sizeof(float) * kcache_size);                                                                                      
    cudaMalloc((void **)&d_kcache, sizeof(float) * kcache_size);                                                                                  
    float *h_vcache;                                                                                                                              
    float *d_vcache;                                                                                                                              
    int vcache_size = max_seq_len * batch_size * kv_num_heads * head_size;                                                                        
    h_vcache = (float *)malloc(sizeof(float) * vcache_size);                                                                                      
    cudaMalloc((void **)&d_vcache, sizeof(float) * vcache_size);                                                                                  
    for (int i = 0; i < qkv_size; i++)                                                                                                            
    {                                                                                                                                             
        if (i < batch_size * num_heads * head_size)                                                                                               
        {                                                                                                                                         
            if (i < batch_size * num_heads * head_size / 2)                                                                                       
            {                                                                                                                                     
                h_qkv[i] = (float)(i + 1);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (float)(i - 3) / 10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        else if (i < batch_size * (num_heads + kv_num_heads) * head_size)                                                                         
        {                                                                                                                                         
            if (i < batch_size * (num_heads + kv_num_heads / 2) * head_size)                                                                      
            {                                                                                                                                     
                h_qkv[i] = (float)(i + 5);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (float)(i + 1) / 10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        else if (i < batch_size * (num_heads + kv_num_heads * 2) * head_size)                                                                     
        {                                                                                                                                         
            if (i < batch_size * (num_heads + kv_num_heads + kv_num_heads / 2) * head_size)                                                       
            {                                                                                                                                     
                h_qkv[i] = (float)(i - 3);                                                                                                        
            }                                                                                                                                     
            else                                                                                                                                  
            {                                                                                                                                     
                h_qkv[i] = (float)(i - 7) / 10;                                                                                                   
            }                                                                                                                                     
        }                                                                                                                                         
        printf("h_qkv[%d]= %f \n", i, h_qkv[i]);                                                                                                  
    }                                                                                                                                             
    float *h_q = h_qkv;                                                                                                                           
    float *h_k = h_q + batch_size * num_heads * head_size;                                                                                        
    float *h_v = h_k + batch_size * (kv_num_heads + num_heads) * head_size;                                                                       
    for (int i = 0; i < (kcache_size * h_step) / max_seq_len; i++)                                                                                
    {                                                                                                                                             
        if (i < kcache_size / 2)                                                                                                                  
        {                                                                                                                                         
            h_kcache[i] = (float)(i + 1);                                                                                                         
            h_vcache[i] = (float)(i + 1);                                                                                                         
        }                                                                                                                                        
        else                                                                                                                                     
        {                                                                                                                                        
            h_kcache[i] = (float)(i - kcache_size / 2 + 1) / 10;                                                                                 
            h_vcache[i] = (float)(i - kcache_size / 2 + 1) / 10;                                                                                 
        }                                                                                                                                        
        printf("h_kcache[%d]= %f\n", i, h_kcache[i]);                                                                                           
        printf("h_vcache[%d]= %f\n", i, h_vcache[i]);                                                                                           
    }                                                                                                                                            
    float *h_o;                                                                                                                                  
    float *d_o;                                                                                                                                  
    int o_size = batch_size * num_heads * head_size;                                                                                             
    h_o = (float *)malloc(sizeof(float) * o_size);                                                                                               
    cudaMalloc((void **)&d_o, sizeof(float) * o_size);                                                                                           
    bool *h_finished = (bool *)malloc(sizeof(bool) * batch_size);                                                                                
    bool *d_finished;                                                                                                                            
    cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size);                                                                                 
    for (int i = 0; i < batch_size; i++)                                                                                                         
    {                                                                                                                                            
        h_finished[i] = static_cast<bool>(0);                                                                                                    
    }                                                                                                                                            
    float *h_qkv_bias = (float *)malloc(sizeof(float) * (2 * kv_num_heads + num_heads) * head_size);                                             
    float *d_qkv_bias;                                                                                                                           
    cudaMalloc((void **)&d_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size);                                                
    for (int i = 0; i < (2 * kv_num_heads + num_heads) * head_size; i++)                                                                         
    {                                                                                                                                            
        h_qkv_bias[i] = (float)0.0f;                                                                                                             
    }                                                                                                                                            
    cudaMemcpy(d_qkv, h_qkv, sizeof(float) * batch_size * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);                   
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * (2 * kv_num_heads + num_heads) * head_size, cudaMemcpyHostToDevice);                      
    cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice);                                                       
    cudaMemcpy(d_kcache, h_kcache, sizeof(float) * kcache_size, cudaMemcpyHostToDevice);                                                         
    cudaMemcpy(d_vcache, h_vcache, sizeof(float) * vcache_size, cudaMemcpyHostToDevice);                                                         
    DataType type = getTensorType<float>();                                                                                                      
    DataType type_bool = getTensorType<bool>();                                                                                                  
    DataType type_int = getTensorType<int>();                                                                                                    
    TensorWrapper<float> *qkv = new TensorWrapper<float>(GPU, type, {batch_size, num_heads + 2 * kv_num_heads, head_size}, d_qkv);               
    TensorWrapper<float> *kcache = new TensorWrapper<float>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_kcache);
    TensorWrapper<float> *vcache = new TensorWrapper<float>(GPU, type, {h_layer_id, batch_size, kv_num_heads, max_seq_len, head_size}, d_vcache);
    TensorWrapper<bool> *finished = new TensorWrapper<bool>(GPU, type_bool, {batch_size}, d_finished);                                           
    TensorWrapper<int> *step = new TensorWrapper<int>(CPU, type_int, {1}, &h_step);                                                              
    TensorWrapper<int> *layer_id = new TensorWrapper<int>(CPU, type_int, {1}, &h_layer_id);                                                      
    TensorWrapper<float> *mha_output = new TensorWrapper<float>(GPU, type, {batch_size, num_heads, head_size}, d_o);                             
    BaseWeight<float> qkv_weight;                                                                                                                
    qkv_weight.bias = d_qkv_bias;                                                                                                                
    LLaMAAttentionStaticParams params;                                                                                                           
    params.rotary_embedding_dim = rotary_embedding_dim;                                                                                          
    params.rotary_embedding_base = rotary_embedding_base;                                                                                        
    params.max_position_embeddings = max_position_embeddings;                                                                                    
    params.use_dynamic_ntk = false;                                                                                                              
    launchDecoderMaskedMHA(qkv, qkv_weight, layer_id, kcache, vcache, finished, step, mha_output, params);                                       
    CHECK(cudaMemcpy(h_o, d_o, sizeof(float) * o_size, cudaMemcpyDeviceToHost));                                                                 
    float *CPU_output = (float *)malloc(sizeof(float) * o_size);                                                                                 
    CPUMaskedAttn<float>(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, head_size, h_step);                               
    bool is_true = CheckResult<float>(CPU_output, h_o, o_size);                                                                                  
    if (is_true)                                                                                                                                 
    {                                                                                                                                            
        printf("test passed\n");                                                                                                                   
    }                                                                                                                                            
    else                                                                                                                                         
    {                                                                                                                                            
        printf("test failed\n");                                                                                                                   
    }                                                                                                                                            
    free(h_qkv);                                                                                                                                 
    free(h_kcache);                                                                                                                              
    free(h_vcache);                                                                                                                              
    free(h_o);                                                                                                                                   
    free(CPU_output);                                                                                                                            
    free(h_finished);                                                                                                                            
    cudaFree(d_finished);                                                                                                                        
    cudaFree(d_qkv);                                                                                                                             
    cudaFree(d_o);                                                                                                                               
    cudaFree(d_kcache);                                                                                                                          
    cudaFree(d_vcache);

}
