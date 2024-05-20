#include <stdio.h>
#include "src/kernels/input_embedding.h"
#include "src/utils/cuda_debug_utils.cuh"
template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
               T* output, 
               const T* embed_table,
               const int max_context_token_num,
               const int hidden_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < max_context_token_num * hidden_size) {
        int id = input_ids[index / hidden_size];
        output[index] = embed_table[id * hidden_size + index % hidden_size];
        index += blockDim.x * gridDim.x;
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [token num]
                          TensorWrapper<T>* output,       // FP32 [token num, hidden_size] = [token num, 4096]
                          EmbeddingWeight<T>* embed_table// FP32 [vocal_size, hidden_size]
                          ) {
    const int blockSize = 256;
    const int max_context_token_num = output->shape[0]; // token num
    const int hidden_size = output->shape[1];
    const int gridSize = 2048;
    LLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");
    embeddingFunctor<T><<<gridSize, blockSize>>>(input_ids->data,
                                                 output->data,
                                                 embed_table->data,
                                                 max_context_token_num,
                                                 hidden_size);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(output->data);
#else
#endif
}

template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table);
