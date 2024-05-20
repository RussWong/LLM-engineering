#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template <typename T>
void launchConcatKVCache(TensorWrapper<T> *k_src, // from qkv bias and rope
                          TensorWrapper<T> *v_src,
                          TensorWrapper<int> *layer_id,         // layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                          TensorWrapper<int> *cur_query_length, // current epoch or local input length,[batchsize]
                          TensorWrapper<int> *history_length,
                          TensorWrapper<T> *k_dst,
                          TensorWrapper<T> *v_dst); // (RussWong)note: 少写一个;都会发生很多奇怪的错误
