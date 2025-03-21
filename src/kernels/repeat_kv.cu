#include "src/kernels/repeat_kv.h"
#include "src/utils/cuda_debug_utils.cuh"
#include <iostream>
// if MQA or GQA, we should use this transpose to broadcast kv head num to q head num
//[num layers, bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
// context_length.shape=[bs]
// bugs1: when k_dst.shape = [1,32,13,128],现在这个k_dst以13*128为单位循环第一个13*128的值
// solu1: launcher函数里面获取kv cache的shape出错，需要仔细核对各个TensorWrapper的shape再通过正确索引获取
template <typename T>
__global__ void repeat_value_cache(T *v_dst,
                                   const T *v_src,
                                   const size_t layer_offset,
                                   const int head_num,
                                   const int q_head_per_kv,
                                   const int head_size,
                                   const int *context_length,
                                   const int max_k_len,
                                   const int max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id = blockIdx.z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto val_src = v_src + layer_offset;
    const auto val_dst = v_dst;

    const auto seq_len = context_length[batch_id];

    const int v_head_size_id = idx % head_size;
    const int v_seq_len_id = idx / head_size;
    // only fetch context_length(<max_seq_len) kv data from all kv cache of current seq
    if (v_seq_len_id < seq_len)
    {
        const int64_t src_idx = batch_id * (head_num / q_head_per_kv) * head_size * max_seq_len + // B
                                head_id / q_head_per_kv * head_size * max_seq_len +               // H
                                v_seq_len_id * head_size +                                        // s
                                v_head_size_id;                                                   // D/x

        const int64_t dst_idx = batch_id * head_num * head_size * max_k_len + // B
                                head_id * head_size * max_k_len +             // H
                                v_seq_len_id * head_size +                    // s
                                v_head_size_id;                               // D/x

        val_dst[dst_idx] = val_src[src_idx];
    }
}
template <typename T>
void launchRepeatKVCache(TensorWrapper<T> *k_cache_src, //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<T> *v_cache_src, //{num_layers, batch_size, kv_head_num, max_seq_len, head_size}
                         TensorWrapper<int> *context_length,
                         TensorWrapper<int> *layer_id,
                         TensorWrapper<T> *k_cache_dst, //{batch_size, head_num, max_k_len, head_size}
                         TensorWrapper<T> *v_cache_dst)
{
    int batch_size = context_length->shape[0];
    int kv_head_num = k_cache_src->shape[2]; // (RussWong)note: we should carefully access the shape value, corresponding to the place where tensorwapper is defined
    int max_seq_len = k_cache_src->shape[3];
    int head_num = k_cache_dst->shape[1];

    int max_k_len = k_cache_dst->shape[2];
    int head_size = k_cache_dst->shape[3];
    int layer = layer_id->getVal();
    // (RussWong)note: if layer id is on GPU, here MUSTN'T use layer_id->getVal<int>(), because we cant access GPU memory directly by [] if data is on GPU
    // (RussWong)note: so we can make layer data locate on CPU, so that we can access data by []
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    int q_head_per_kv = head_num / kv_head_num;
    int blockSize = 128;
    dim3 block(blockSize);
    dim3 grid((max_k_len * head_size + blockSize - 1) / blockSize, batch_size, head_num);
    repeat_value_cache<T><<<grid, block>>>(v_cache_dst->data,
                                           v_cache_src->data,
                                           layer_offset,
                                           head_num,
                                           q_head_per_kv,
                                           head_size,
                                           context_length->data,
                                           max_k_len,
                                           max_seq_len);

    repeat_value_cache<T><<<grid, block>>>(k_cache_dst->data,
                                           k_cache_src->data,
                                           layer_offset,
                                           head_num,
                                           q_head_per_kv,
                                           head_size,
                                           context_length->data,
                                           max_k_len,
                                           max_seq_len);
#ifdef PRINT_DATA
    printf("repeat kv kernel top2 result:\n");
    print_data<<<1, 1>>>(k_cache_dst->data);
#else
#endif
}

template void launchRepeatKVCache(TensorWrapper<float> *k_cache_src,
                                  TensorWrapper<float> *v_cache_src,
                                  TensorWrapper<int> *context_length,
                                  TensorWrapper<int> *layer_id,
                                  TensorWrapper<float> *k_cache_dst,
                                  TensorWrapper<float> *v_cache_dst);
template void launchRepeatKVCache(TensorWrapper<half> *k_cache_src,
                                  TensorWrapper<half> *v_cache_src,
                                  TensorWrapper<int> *context_length,
                                  TensorWrapper<int> *layer_id,
                                  TensorWrapper<half> *k_cache_dst,
                                  TensorWrapper<half> *v_cache_dst);
