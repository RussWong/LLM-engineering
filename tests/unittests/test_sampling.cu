#include <iostream>
#include "src/kernels/sampling.h"
#include "src/utils/macro.h"
// (RussWong)note: there is no CPU implementation of this kernel
// we compare the kernel correctnesss by eyes and result print infos
// `./test_sampling` to test fp32 GPU kernel
// `./test_sampling 1` to test fp16 GPU kernel
#define LAUNCH_SAMPLING(dtype)                                                                                 \
    int *h_topkid;                                                                                             \
    int *d_topkid;                                                                                             \
    h_topkid = (int *)malloc(sizeof(int) * batch_size * K);                                                    \
    cudaMalloc((void **)&d_topkid, sizeof(int) * batch_size * K);                                              \
    dtype *h_topkval;                                                                                          \
    dtype *d_topkval;                                                                                          \
    h_topkval = (dtype *)malloc(sizeof(dtype) * batch_size * K);                                               \
    cudaMalloc((void **)&d_topkval, sizeof(dtype) * batch_size * K);                                           \
    int *h_outid;                                                                                              \
    int *d_outid;                                                                                              \
    h_outid = (int *)malloc(sizeof(int) * batch_size);                                                         \
    cudaMalloc((void **)&d_outid, sizeof(int) * batch_size);                                                   \
    int *h_cuseqlen;                                                                                           \
    int *d_cuseqlen;                                                                                           \
    h_cuseqlen = (int *)malloc(sizeof(int) * batch_size);                                                      \
    cudaMalloc((void **)&d_cuseqlen, sizeof(int) * batch_size);                                                \
    bool *h_finished;                                                                                          \
    bool *d_finished;                                                                                          \
    h_finished = (bool *)malloc(sizeof(bool) * batch_size);                                                    \
    cudaMalloc((void **)&d_finished, sizeof(bool) * batch_size);                                               \
    for (int i = 0; i < batch_size; i++)                                                                       \
    {                                                                                                          \
        h_finished[i] = 0;                                                                                     \
        h_cuseqlen[i] = 4;                                                                                     \
    }                                                                                                          \
    for (int i = 0; i < batch_size * K; i++)                                                                   \
    {                                                                                                          \
        h_topkid[i] = i;                                                                                       \
        h_topkval[i] = (dtype)(K - 1 - (i % K));                                                               \
    }                                                                                                          \
    CHECK(cudaMemcpy(d_topkval, h_topkval, sizeof(dtype) * batch_size * K, cudaMemcpyHostToDevice));           \
    DataType type = getTensorType<dtype>();                                                                    \
    TensorWrapper<dtype> *topk_val = new TensorWrapper<dtype>(Device::GPU, type, {batch_size, K}, d_topkval);  \
    CHECK(cudaMemcpy(d_topkid, h_topkid, sizeof(int) * batch_size * K, cudaMemcpyHostToDevice));               \
    CHECK(cudaMemcpy(d_cuseqlen, h_cuseqlen, sizeof(int) * batch_size, cudaMemcpyHostToDevice));               \
    CHECK(cudaMemcpy(d_finished, h_finished, sizeof(bool) * batch_size, cudaMemcpyHostToDevice));              \
    DataType type_int = getTensorType<int>();                                                                  \
    DataType type_bool = getTensorType<bool>();                                                                \
    TensorWrapper<int> *topk_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, K}, d_topkid);    \
    TensorWrapper<int> *cuseqlen = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_cuseqlen);    \
    TensorWrapper<bool> *finished = new TensorWrapper<bool>(Device::GPU, type_bool, {batch_size}, d_finished); \
    TensorWrapper<int> *output_id = new TensorWrapper<int>(Device::GPU, type_int, {batch_size}, d_outid);      \
    IntDict intParams;                                                                                         \
    intParams.insert({"step", step});                                                                          \
    intParams.insert({"vocab_size", vocab_size});                                                              \
    intParams.insert({"end_id", end_id});                                                                      \
    std::cout << "before launch sampling kernel" << std::endl;                                                 \
    launchSampling<dtype>(topk_id, topk_val, cuseqlen, finished, output_id, intParams);                               \
    std::cout << "after launch sampling kernel" << std::endl;                                                  \
    std::cout << "cuda memcpy device to host" << std::endl;                                                    \
    CHECK(cudaMemcpy(h_outid, output_id->data, sizeof(int) * batch_size, cudaMemcpyDeviceToHost));              \
    for (int i = 0; i < batch_size; i++)                                                                       \
    {                                                                                                          \
        std::cout << "seq" << i + 1 << ":" << h_outid[i] << std::endl;                                         \
    }                                                                                                          \
    free(h_topkid);                                                                                            \
    free(h_topkval);                                                                                           \
    free(h_finished);                                                                                          \
    free(h_cuseqlen);                                                                                          \
    free(h_outid);                                                                                             \
    cudaFree(d_topkid);                                                                                        \
    cudaFree(d_topkval);                                                                                       \
    cudaFree(d_finished);                                                                                      \
    cudaFree(d_cuseqlen);                                                                                      \
    cudaFree(d_outid);

int main(int argc, char *argv[])
{
    const int batch_size = 3;
    const int K = 3;
    int vocab_size = 1000;
    int step = 6;
    int end_id = 10;

    if (argv[1])
    {
        LAUNCH_SAMPLING(half)    
    } else {
        LAUNCH_SAMPLING(float)    
    }
}
