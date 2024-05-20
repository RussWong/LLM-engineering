#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>
#include "src/kernels/topK.h"
// (RussWong)note:
// there is no top k cpu kernel implementation now
// we compare the kernel correctnesss by eyes and result print infos
// `./test_topk` to test fp32 GPU kernel
int main() {
    const int batch_size = 1;
    const int vocab_size = 30000;
    const int beamwidth = 2;
    const int K = 5;
    const int BlockPerBeam = 8;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int probs_size = batch_size * vocab_size * beamwidth;
    float* h_probs;
    float *d_probs;
    h_probs = (float*)malloc(sizeof(float) * probs_size);
    cudaMalloc((void**)&d_probs, sizeof(float) * probs_size);
    
    int topK_val_buf_size = batch_size * beamwidth * BlockPerBeam * K;
    int topK_ids_buf_size = batch_size * beamwidth * BlockPerBeam * K;
    int final_topK_val_buf_size = batch_size * beamwidth * K; // sampling topK buf size, beamsearch topK size = [batch_size * beam_width * beam_width]


    int *d_tmp_topk_ids;
    cudaMalloc((void**)&d_tmp_topk_ids, sizeof(int) * topK_ids_buf_size);

    float *d_tmp_topk_vals;
    cudaMalloc((void**)&d_tmp_topk_vals, sizeof(float) * topK_val_buf_size);

    int* h_final_topk_ids;
    int *d_final_topk_ids;
    h_final_topk_ids = (int*)malloc(sizeof(int) * final_topK_val_buf_size);
    cudaMalloc((void**)&d_final_topk_ids, sizeof(int) * final_topK_val_buf_size);

    float* h_final_topk_vals;
    float *d_final_topk_vals;
    h_final_topk_vals = (float*)malloc(sizeof(float) * final_topK_val_buf_size);
    cudaMalloc((void**)&d_final_topk_vals, sizeof(float) * final_topK_val_buf_size);

    for(int i = 0; i < probs_size; i++) { // 0-59999
       h_probs[i] = i;
    }
    cudaMemcpy(d_probs, h_probs, sizeof(float)*probs_size, cudaMemcpyHostToDevice);

    DataType type_float = getTensorType<float>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<float>* probs_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                type_float,
                                                                {batch_size * beamwidth, vocab_size}, 
                                                                d_probs);
    TensorWrapper<int> *tmp_topk_ids = new TensorWrapper<int>(Device::GPU, 
                                                                type_int,
                                                                {batch_size, beamwidth, BlockPerBeam, K}, 
                                                                d_tmp_topk_ids);
    TensorWrapper<float>* tmp_topk_vals = new TensorWrapper<float>(Device::GPU, 
                                                                type_float,
                                                                {batch_size, beamwidth, BlockPerBeam, K}, 
                                                                d_tmp_topk_vals);
    TensorWrapper<int> *final_topk_ids = new TensorWrapper<int>(Device::GPU, 
                                                                type_int,
                                                                {batch_size * beamwidth, K}, 
                                                                d_final_topk_ids);
    TensorWrapper<float> *final_topk_vals = new TensorWrapper<float>(Device::GPU, 
                                                                type_float,
                                                                {batch_size * beamwidth, K}, 
                                                                d_final_topk_vals);
    // debug info, better to retain: std::cout << "before launch kernel" << std::endl;
    launchTopKforBeamSearch(probs_tensor, tmp_topk_ids, tmp_topk_vals, final_topk_ids, final_topk_vals);
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_final_topk_ids, d_final_topk_ids, sizeof(int) * final_topK_val_buf_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_final_topk_vals, d_final_topk_vals,  sizeof(float) * final_topK_val_buf_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < final_topK_val_buf_size; i++) {
        int id = h_final_topk_ids[i];
        printf("topK id = %d\n", id);
        float val = h_final_topk_vals[i];
        printf("topK val =%f\n", val);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_probs);
    free(h_final_topk_ids);
    free(h_final_topk_vals);
    cudaFree(d_probs);
    cudaFree(d_final_topk_ids);
    cudaFree(d_final_topk_vals);
    cudaFree(d_tmp_topk_ids);
    cudaFree(d_tmp_topk_vals);
}
