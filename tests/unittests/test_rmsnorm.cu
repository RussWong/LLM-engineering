#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <iostream>
#include "src/kernels/rmsnorm_kernel.h"

#include <stdio.h>
// (RussWong)note: this kernel's CPU implementation is absolutely right.
// But when you are implementing LLMs inference on CPU, I dont recommend to reuse the CPU kernel, because its performance is bad
// `./test_residual` to test fp32 GPU kernel
// `./test_residual 1` to test fp16 GPU kernel

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

void CPUfusedresidandRMSNorm(float* h_decoder_out, 
                                    float* h_scale, float eps, int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        float input = 0.0f;
        float sum = 0.0f;
	for (int i = 0; i < hidden_units; i++) {
            input = h_decoder_out[b * hidden_units + i];
	    sum += input * input;
        }
        mean = (float)sum / hidden_units;
        inv_fenmu = rsqrt(mean + eps);
        
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
        }
    }
}

template<typename T>
bool CheckResult(float* CPUoutput, T* GPUoutput, int output_size) {
    float fp32GPUoutput = 0.0f;
    for(int i = 0; i < output_size; i++) {
        fp32GPUoutput = (float)GPUoutput[i];
        if(fabs(CPUoutput[i] - fp32GPUoutput) > 1e-6){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], fp32GPUoutput);
            return false;
        }

    }
    return true;
}

int main(int argc, char *argv[]) {
    const int num_tokens = 64;
    const int hidden_units = 4096;
    const int total_size = num_tokens * hidden_units;
    float eps = 1e-6;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    // first param = true or 1, we go fp32
    if (argv[1]) {
        half* h_decoder_out = (half*) malloc(sizeof(half) * total_size);
        half* decoder_out = (half*) malloc(sizeof(half) * total_size);
        half* d_decoder_out;
        cudaMalloc((void**)&d_decoder_out, sizeof(half) * total_size);
        for(int i = 0; i < total_size; i++) { 
            h_decoder_out[i] = 1.0f;
        }
        // to save residual used by fusedResidualAndRmsnorm
        half* d_decoder_rsd;
        cudaMalloc((void**)&d_decoder_rsd, sizeof(half) * total_size);
        //rmsnorm weights
        half* h_scale = (half*) malloc(sizeof(half) * hidden_units);
        half* d_scale;
        cudaMalloc((void**)&d_scale, sizeof(half) * hidden_units);
        for(int i = 0; i < hidden_units; i++) { 
            h_scale[i] = (half)1;
        }

        CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(half) * total_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_scale, h_scale, sizeof(half) * hidden_units, cudaMemcpyHostToDevice));

        DataType type_half = getTensorType<half>();
        DataType type_int = getTensorType<int>();
        TensorWrapper<half>* decoder_out_tensor = new TensorWrapper<half>(Device::GPU, 
                                                                            type_half,
                                                                            {num_tokens, hidden_units}, 
                                                                            d_decoder_out);
        TensorWrapper<half>* decoder_rsd= new TensorWrapper<half>(Device::GPU, 
                                                                            type_half,
                                                                            {num_tokens, hidden_units}, 
                                                                            d_decoder_rsd);

        LayerNormWeight<half> scale;
        scale.gamma = d_scale;
        // debug info, better to retain: 
        std::cout << "before launch kernel" << std::endl;
        launchRMSNorm(decoder_out_tensor, decoder_rsd, scale, eps);
        // debug info, better to retain: 
        std::cout << "after launch kernel" << std::endl;
        // debug info, better to retain: 
        std::cout << "cuda memcpy device to host" << std::endl;
        // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
        CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(half) * total_size, cudaMemcpyDeviceToHost));
        
        float* CPUout = (float*) malloc(sizeof(float) * total_size);
        for(int i = 0; i < total_size; i++){
            CPUout[i] = 1.0f;
        }
        float* cpu_scale = (float*) malloc(sizeof(float) * hidden_units);
        for(int i = 0; i < hidden_units; i++) { 
            cpu_scale[i] = (float)1;
        }
        CPUfusedresidandRMSNorm(CPUout, cpu_scale, eps, hidden_units, num_tokens);
        bool is_right = CheckResult<half>(CPUout, decoder_out, total_size);
        // debug info, better to retain: 
        std::cout << "rmsnorm passed" << std::endl;
        free(h_decoder_out);
        free(h_scale);
        free(cpu_scale);
        free(CPUout);
        free(decoder_out);
        cudaFree(d_decoder_out);
        cudaFree(d_scale);
    } else {
        float* h_decoder_out = (float*) malloc(sizeof(float) * total_size);
        float* decoder_out = (float*) malloc(sizeof(float) * total_size);
        float* d_decoder_out;
        cudaMalloc((void**)&d_decoder_out, sizeof(float) * total_size);
        for(int i = 0; i < total_size; i++) { 
            h_decoder_out[i] = (float)(i % 2 + 1);
        }
        // to save residual used by fusedResidualAndRmsnorm
        float* d_decoder_rsd;
        cudaMalloc((void**)&d_decoder_rsd, sizeof(float) * total_size);
        //rmsnorm weights
        float* h_scale = (float*) malloc(sizeof(float) * hidden_units);
        float* d_scale;
        cudaMalloc((void**)&d_scale, sizeof(float) * hidden_units);
        for(int i = 0; i < hidden_units; i++) { 
            h_scale[i] = (float)(i % 2 + 1);
        }

        CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_scale, h_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

        DataType type_float = getTensorType<float>();
        DataType type_int = getTensorType<int>();
        TensorWrapper<float>* decoder_out_tensor = new TensorWrapper<float>(Device::GPU, 
                                                                            type_float,
                                                                            {num_tokens, hidden_units}, 
                                                                            d_decoder_out);
        TensorWrapper<float>* decoder_rsd= new TensorWrapper<float>(Device::GPU, 
                                                                            type_float,
                                                                            {num_tokens, hidden_units}, 
                                                                            d_decoder_rsd);

        LayerNormWeight<float> scale;
        scale.gamma = d_scale;
        // debug info, better to retain: 
        std::cout << "before launch kernel" << std::endl;
        launchRMSNorm(decoder_out_tensor, decoder_rsd, scale, eps);
        // debug info, better to retain: 
        std::cout << "after launch kernel" << std::endl;
        // debug info, better to retain: 
        std::cout << "cuda memcpy device to host" << std::endl;
        // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
        CHECK(cudaMemcpy(decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));
        // 以下float不用被half替换
        float* CPUout = (float*) malloc(sizeof(float) * total_size);
        for(int i = 0; i < total_size; i++){
            CPUout[i] = (float)(i % 2 + 1);
        }
        float* cpu_scale = (float*) malloc(sizeof(float) * hidden_units);
        for(int i = 0; i < hidden_units; i++) { 
            cpu_scale[i] = (float)(i % 2 + 1);
        }
        CPUfusedresidandRMSNorm(CPUout, cpu_scale, eps, hidden_units, num_tokens);
        bool is_right = CheckResult<float>(CPUout, decoder_out, total_size);
        // debug info, better to retain: 
        std::cout << "rmsnorm passed" << std::endl;
        free(h_decoder_out);
        free(h_scale);
        free(cpu_scale);
        free(CPUout);
        free(decoder_out);
        cudaFree(d_decoder_out);
        cudaFree(d_scale);
    }
}

