#include <iostream>
#include "src/kernels/sampling.h"
// mini-softmax + curand_sample
// input: [bs, K] from topK output
// output: [bs]
// (Russwong)note: beamsearch不存在sampling，所以bsxbm = bs
template<typename T>
__global__ void SamplingKernel(int* topk_id,
                               T* topk_val, //[bs, K] from topK
                               int* output_id, //[bs]
                               int* seqlen, //cumulated seq len,[bs]
                               bool* is_finished, //[bs]
                               int K,
                               int rand_num, // step
                               int end_id, // when initialize llama model, we will init it, and this is a fixed val
                               int vocab_size)
{
    int batch_id = blockIdx.x;
    int bid = batch_id;
    int tid = threadIdx.x;
    int offset = batch_id * K + tid;
    T max_val = topk_val[batch_id * K]; // max val is the top of the buffer, because topK
    topk_val[offset] = (T)(expf((float)topk_val[offset] - (float)max_val));
    __shared__ float thredhold, sum;
    if(tid == 0) {
        sum = 0.0f;
        for(int i = 0; i < K; i++) {
            sum += (float)topk_val[batch_id * K + i];
        }
        curandState_t state;
        // (Russwong)note: curand_init API only support ulonglong data type
        curand_init((unsigned long long)rand_num,(unsigned long long)bid, (unsigned long long)0, &state);
        thredhold = (float)curand_uniform(&state) * sum; // for a block
        output_id[bid] = topk_id[bid * K] % vocab_size; 
        for(int i = 0; i < K; i++) {
            thredhold = thredhold - (float)topk_val[batch_id * K + i];
            if(thredhold < 0) {
                output_id[bid] = topk_id[batch_id * K + i] % vocab_size;
                break;
            }
        }
        seqlen[bid] = is_finished[bid] ? seqlen[bid] : seqlen[bid] + 1;
        is_finished[bid] = output_id[bid] == end_id ? 1 : 0;
    }
}

template<typename T>
void launchSampling(TensorWrapper<int>* topk_id,
                    TensorWrapper<T>* topk_val,
                    TensorWrapper<int>* seqlen,
                    TensorWrapper<bool>* is_finished,
                    TensorWrapper<int>* output_id,
                    IntDict& params) {
    int batch_size = topk_id->shape[0];
    int K = topk_id->shape[1];
    int vocab_size = params["vocab_size"];
    int step = params["step"];
    int end_id = params["end_id"];

    dim3 grid(batch_size);
    dim3 block(K); // K is small, so directly allocate K threads is enough
    SamplingKernel<<<grid, block>>>(
        topk_id->data,
        topk_val->data,
        output_id->data,
        seqlen->data,
        is_finished->data,
        K,
        step,
        end_id,
        vocab_size
    );
}

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<float>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);

template void launchSampling(TensorWrapper<int>* topk_id,
                            TensorWrapper<half>* topk_val,
                            TensorWrapper<int>* seqlen,
                            TensorWrapper<bool>* is_finished,
                            TensorWrapper<int>* output_id,
                            IntDict& params);
