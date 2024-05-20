#include <math.h>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
#include "src/layers/attention/context_attention.h"
//(RussWong) note: layers文件夹下，很多操作后面我都加了`DeviceSyncAndCheckCudaError();`，大家可手动删除或者按照lesson30所示添加条件编译代码
template<typename T>
LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(
                               int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator), 
    hidden_units(head_num * head_size),
    attn_static_params(attn_params),
    // TODO: check kv_head_num is divided by haed_num
    q_head_per_kv(head_num / kv_head_num),
    scale(float(1 / sqrt(head_size))){}

template<typename T>    
void LLaMAContextAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams& params) {
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    // for qkv linear and bias rope
    qkv_buf_wo_pad = new TensorWrapper<T>(Device::GPU, type, {num_tokens, qkv_head_num,  head_size});
    q_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size}); //pad to max_q_len taht is max len of cur batch
    k_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}); //why here isn't max_k_len?cause the q/k/v is got by {bs, q_len, hiddenunits} * {hiddenunits, hiddenunits}
    v_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
    // for transpose kv cache
    k_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});// why not kv_head_num？need repeat kv to adapt q head num
    v_cache_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});// max k len mean max context len in this batch
    // for q*k and softmax
    qk_buf = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});
    //qk * v
    qkv_buf_w_pad = new TensorWrapper<T>(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    //remove padding
    qkv_buf_wo_pad_1 = new TensorWrapper<T>(Device::GPU, type, {num_tokens, head_num, head_size});
    
    qkv_buf_wo_pad->data = allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T) * num_tokens * qkv_head_num * head_size, false);
    q_buf_w_pad->data = allocator->Malloc(
        q_buf_w_pad->data, sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size, false);
    k_buf_w_pad->data = (T*)q_buf_w_pad->data + head_num * batch_size * max_q_len * head_size;
    v_buf_w_pad->data = (T*)k_buf_w_pad->data + kv_head_num * batch_size * max_q_len * head_size;
    k_cache_buf->data = allocator->Malloc(
        k_cache_buf->data, 2 * sizeof(T) * batch_size * head_num * max_k_len * head_size, false);
    v_cache_buf->data = (T*)k_cache_buf->data + batch_size * head_num * max_k_len * head_size;
    // store qk and inplace store softmax output
    qk_buf->data =
        allocator->Malloc(qk_buf->data, sizeof(T) * batch_size * head_num * max_q_len * max_k_len, false);
    // store qk*v
    qkv_buf_w_pad->data = allocator->Malloc(
        qkv_buf_w_pad->data, sizeof(T) * batch_size * max_q_len * head_num * head_size, false);
    qkv_buf_wo_pad_1->data= allocator->Malloc(qkv_buf_wo_pad_1->data, sizeof(T) * num_tokens * head_num * head_size, false);
}

template<typename T>    
void LLaMAContextAttentionLayer<T>::freeBuf(){
    allocator->Free(qkv_buf_wo_pad->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(q_buf_w_pad->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(k_cache_buf->data);
    DeviceSyncAndCheckCudaError();
//    allocator->Free(v_cache_buf->data); // (RussWong) note: no need to free v cache buf, because its included in k cache buf->data
//    DeviceSyncAndCheckCudaError();
    allocator->Free(qk_buf->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qkv_buf_w_pad->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(qkv_buf_wo_pad_1->data);
}
// (RussWong) note: params order of launcher function in LaMAContextAttentionLayer<T>::forward: (input[Tensor], input[Tensor],...,weight[Weight], output[*])
template<typename T>
void LLaMAContextAttentionLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params)
{   
    // RussWong) note: allocate intermediat buf of the layer forward
    allocForForward(params);
    //1.qkv linear
    //shape:[num_tokens, qhiddenunits] * [qhiddenunits, hiddenunits]
    Tensor* attention_input = inputs["attention_input"];
    launchLinearGemm(attention_input->as<T>(), weights.qkv, qkv_buf_wo_pad, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    //2.qkv bias and rope and padding
    //shape:[num_tokens, hiddenunits]=>{batch_size, q(kv)head_num, max_q_len, head_size}
    //(RussWong) note: qkv bias is not existed in llama
    Tensor* padding_offset = inputs["padding_offset"];
    Tensor* history_length = inputs["history_length"];
    Tensor* input_length = inputs["input_length"];
    Tensor* layer_id = inputs["layer_id"]; //ON CPU
    launchAddFusedQKVBiasTransposeAndRoPE(q_buf_w_pad, k_buf_w_pad, v_buf_w_pad, qkv_buf_wo_pad,
                                        weights.qkv, padding_offset->as<int>(), history_length->as<int>(), input_length->as<int>(), static_params);
#ifndef PERF
    DeviceSyncAndCheckCudaError();
#else
#endif
#ifdef SAVE_DATA
    save_tensor(q_buf_w_pad ,"q_buf_after_rope.bin", layer_id->as<int>()); //{batch_size, head_num, max_q_len, head_size}
#else
#endif
    //3.concat past kv cache
    // max q len is input length with bs = 1
    //shape:{batch_size, kv_head_num, max_q_len, headsize}=>(num_layer ,batch , maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], hidden_units_}; 
    Tensor* all_k_cache = outputs["all_k_cache"];
    Tensor* all_v_cache = outputs["all_v_cache"];
    launchConcatKVCache(k_buf_w_pad, v_buf_w_pad, layer_id->as<int>(), input_length->as<int>(), history_length->as<int>(), all_k_cache->as<T>(), all_v_cache->as<T>());
    DeviceSyncAndCheckCudaError();
    //4.MHA/MQA/GQA part, reduce kv cache size to [num_layer, bs, kv head num, max_seq_len, head size]
    // 0.kv repeat/broadcast to adapt batchgemm shape requirement([bs, head num, seqlen, head size]) if need
    // shape:[num_layer, bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
    Tensor* context_length = inputs["context_length"];
    launchRepeatKVCache(all_k_cache->as<T>(), all_v_cache->as<T>(), context_length->as<int>(), 
                                layer_id->as<int>(), k_cache_buf, v_cache_buf);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(k_cache_buf ,"k_buf_after_repeat.bin", layer_id->as<int>()); //{batch_size, head_num, max_k_len, head_size}
#else
#endif
    // 1.qk 
    // shape:[bs,qhead,qlen,headsize]*[bs,qhead,klen,headsize](N*T)=>[bs,head,qlen,klen]
    launchLinearStridedBatchGemm(q_buf_w_pad, k_cache_buf, qk_buf, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    // 2.scale+mask+softmax
    Tensor* attention_mask = inputs["attention_mask"];
    launchScaleMaskAndSoftmax(qk_buf, attention_mask->as<T>(), qk_buf, scale);
    DeviceSyncAndCheckCudaError();
    // 3.qk*v 
    // shape:[bs,head,qlen,klen]=>[bs,head,qlen,headsize]
    launchLinearStridedBatchGemm(qk_buf, v_cache_buf, qkv_buf_w_pad, cublas_wrapper, false, false);
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(qkv_buf_w_pad ,"qk_v_buf_after_bmm.bin", layer_id->as<int>()); // {batch_size, head_num, max_q_len, head_size}
#else
#endif
    // 4.transpose+reshape(shape:[bs,head,seqlen,headsize]=>[bs,seqlen,head,headsize]=>[numtokens,hiddenunits])+remove padding
    launchTransposeOutRemovePadding(qkv_buf_w_pad, padding_offset->as<int>(), qkv_buf_wo_pad_1);
    DeviceSyncAndCheckCudaError();
    // 5.output linear 
    // shape:[numtokens,hiddenunits]=>[numtokens,hiddenunits]
    Tensor* attention_output = outputs["attention_output"];
    launchLinearGemm(qkv_buf_wo_pad_1, weights.output, attention_output->as<T>(), cublas_wrapper, false, true);
#ifdef SAVE_DATA
    save_tensor(attention_output->as<T>() ,"out_linear_output.bin", layer_id->as<int>()); // {num_tokens, head_num, head_size}
#else
#endif
    DeviceSyncAndCheckCudaError();
    this->freeBuf();
}

template class LLaMAContextAttentionLayer<float>;
template class LLaMAContextAttentionLayer<half>;
