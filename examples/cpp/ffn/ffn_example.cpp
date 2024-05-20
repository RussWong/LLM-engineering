#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/ffn/ffn.h"

int main(int argc, char** argv)
{
    int head_num = 4;
    int head_size = 8;
    int inter_size = 12;
    int hidden_units = head_num * head_size;
    
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;

    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.num_tokens = 14;  
    std::cout << "start malloc/cudamalloc buffer" << "\n";
    float* h_ffn_input = (float*) malloc(sizeof(float) * hidden_units * attn_dyn_params.num_tokens);
    float* d_ffn_input;
    cudaMalloc((void**)&d_ffn_input, sizeof(float) * hidden_units * attn_dyn_params.num_tokens);
    for(int i = 0; i < hidden_units * attn_dyn_params.num_tokens; i++) { 
       h_ffn_input[i] = (float)(i % 2 + 1);
    }    
    float* h_gate_up = (float*) malloc(sizeof(float) * hidden_units * 2 * inter_size);
    float* d_gate_up;
    cudaMalloc((void**)&d_gate_up, sizeof(float) * hidden_units * 2 * inter_size);
    for(int i = 0; i < hidden_units * 2 * inter_size; i++) { 
       h_gate_up[i] = (float)(i % 2 + 1);
    }  
   //  float* h_up = (float*) malloc(sizeof(float) * hidden_units * inter_size);
   //  float* d_up;
   //  cudaMalloc((void**)&d_up, sizeof(float) * hidden_units * inter_size);
   //  for(int i = 0; i < hidden_units * inter_size; i++) { 
   //     h_up[i] = 1.0f;
   //  }  
    float* h_down = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* d_down;
    cudaMalloc((void**)&d_down, sizeof(float) * hidden_units * inter_size);
    for(int i = 0; i < hidden_units * inter_size; i++) { 
       h_down[i] = (float)(i % 2 + 1);
    }  
    float* d_ffn_output;
    cudaMalloc((void**)&d_ffn_output, sizeof(float) * attn_dyn_params.num_tokens * hidden_units);
    std::cout << "end malloc/cudamalloc buffer and start memcpyh2d" << "\n";
    CHECK(cudaMemcpy(d_ffn_input, h_ffn_input, sizeof(float) * hidden_units * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gate_up, h_gate_up, sizeof(float) * hidden_units * 2 * inter_size, cudaMemcpyHostToDevice));
   //  CHECK(cudaMemcpy(d_up, h_up, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_down, h_down, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    LLaMAFFNWeights<float> ffn_weights;
    ffn_weights.gateAndup.data = d_gate_up;
    ffn_weights.gateAndup.shape = {2 * inter_size, hidden_units};
   //  ffn_weights.up.data = d_up;
   //  ffn_weights.up.shape = {hidden_units, inter_size};
    ffn_weights.down.data = d_down;
    ffn_weights.down.shape = {hidden_units, inter_size};
    TensorWrapper<float>* ffn_input = new TensorWrapper<float>(GPU, 
                                                               type, 
                                                               {attn_dyn_params.num_tokens, hidden_units}, 
                                                               d_ffn_input);
    TensorWrapper<float>* ffn_output = new TensorWrapper<float>(GPU, 
                                                               type, 
                                                               {attn_dyn_params.num_tokens, hidden_units}, 
                                                               d_ffn_output);
    TensorMap ffn_inputs{
        {"ffn_input", ffn_input}
    };
    TensorMap ffn_outputs{
        {"ffn_output", ffn_output}
    };
    std::cout << "initializing ffn layer" << "\n";
    LLaMAFFNLayer<float>* ffn_layer = new LLaMAFFNLayer<float>(head_num,
                                                head_size,
                                                inter_size,
                                                stream,
                                                cublas_wrapper,
                                                allocator);
    std::cout << "start fwd" << "\n";
    ffn_layer->forward(ffn_inputs, ffn_outputs, ffn_weights, attn_dyn_params);
    std::cout << "end fwd" << "\n";
    free(h_ffn_input);  
    free(h_gate_up);  
   //  free(h_up);  
    free(h_down); 
    cudaFree(d_ffn_input);  
    cudaFree(d_gate_up);  
   //  cudaFree(d_up);  
    cudaFree(d_down); 
    cudaFree(d_ffn_output);
}
