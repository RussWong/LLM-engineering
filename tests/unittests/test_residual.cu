#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <iostream>
#include "src/kernels/add_residual.h"

#include <stdio.h>
// (RussWong)note: this kernel's CPU implementation is absolutely right.
// But when you are implementing LLMs inference on CPU, I dont recommend to reuse the CPU kernel, because its performance is bad
// `./test_residual` to test fp32 GPU kernel
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

void CPUresidual(float* h_residual, float* h_decoder_out,  int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] += h_residual[b * hidden_units + i];
        }
    }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    for(int i = 0; i < output_size; i++) {
        if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }

    }
    return true;
}

int main() {
    const int num_tokens = 16;
    const int hidden_units = 4096;
    const int total_size = num_tokens * hidden_units;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_residual;
    float* d_residual;
    h_residual = (float*)malloc(sizeof(float) * total_size);
    cudaMalloc((void**)&d_residual, sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++) { 
       h_residual[i] = (float)(i % 2 + 1);
    }

    float* h_decoder_out = (float*) malloc(sizeof(float) * total_size);
    float* decoder_out = (float*) malloc(sizeof(float) * total_size);
    float* d_decoder_out;
    cudaMalloc((void**)&d_decoder_out, sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++) { 
       h_decoder_out[i] = (float)(i % 2 + 1);
    }

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    DataType type_float = getTensorType<float>();
    TensorWrapper<float>* decoder_out_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                        type_float,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_decoder_out);
    TensorWrapper<float>* residual_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                        type_float,
                                                                        {num_tokens, hidden_units}, 
                                                                        d_residual);                                                                        
    // debug info, better to retain: 
    std::cout << "before launch kernel" << std::endl;
    launchAddResidual(residual_tensor, decoder_out_tensor);
    // debug info, better to retain: 
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain: 
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));
    float* CPUout = (float*) malloc(sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++){
        CPUout[i] = (float)(i % 2 + 1);
    }
    CPUresidual(h_residual, CPUout, hidden_units, num_tokens);
    bool is_right = CheckResult(CPUout, decoder_out, total_size);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "AddResidual kernel passed" << std::endl;
    free(h_residual);
    free(h_decoder_out);
    free(CPUout);
    free(decoder_out);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);
}
