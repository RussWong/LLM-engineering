#include <iostream>
#include "src/layers/ffn/ffn.h"
#include "src/utils/debug_utils.h"
//(RussWong) note: layers文件夹下，很多操作后面我都加了`DeviceSyncAndCheckCudaError();`，大家可手动删除或者按照lesson30所示添加条件编译代码
template<typename T>
LLaMAFFNLayer<T>::LLaMAFFNLayer(int head_num,
                               int head_size,
                               int inter_size,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator):
    head_num(head_num),
    head_size(head_size),
    inter_size(inter_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator),
    hidden_units(head_num * head_size) {}

template<typename T>
void LLaMAFFNLayer<T>::allocForForward(LLaMAAttentionDynParams& params){
    int num_tokens = params.num_tokens;
    DataType type = getTensorType<T>(); 
    SwiGLU_input = new TensorWrapper<T>(Device::GPU, type, {num_tokens, 2, inter_size});
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {num_tokens, inter_size});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * num_tokens * 2 * inter_size, false);
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * num_tokens * inter_size, false);
}
template<typename T>
void LLaMAFFNLayer<T>::allocForForward(int batch_size){
    DataType type = getTensorType<T>(); 
    SwiGLU_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, 2, inter_size});
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, inter_size});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * batch_size * 2 * inter_size, false);
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * batch_size * inter_size, false);
}
template<typename T>
void LLaMAFFNLayer<T>::freeBuf(){
    allocator->Free(SwiGLU_input->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(down_proj_input->data);
    DeviceSyncAndCheckCudaError();
}
template<typename T>
void LLaMAFFNLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAFFNWeights<T>& weights, LLaMAAttentionDynParams& params){
    if (params.num_tokens > 0) {
        allocForForward(params);
    } else {
        allocForForward(params.batch_size);
    }
    Tensor* ffn_input = inputs["ffn_input"];
    Tensor* ffn_output = outputs["ffn_output"];
    count += 1;
    bool is_ctx = params.is_ctx;
#ifdef SAVE_DATA 
    save_tensor(ffn_input->as<T>(), "ffn_input.bin", count);
#else
#endif
    // 1.fusedGateUp proj
    launchLinearGemm(ffn_input->as<T>(), weights.gateAndup, SwiGLU_input, cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    // single up proj linear, deprecated due to fuse gate and up into fusedGateAndup
    // launchLinearGemm(ffn_input->as<T>(), weights.up, SwiGLU_input, cublas_wrapper, false, false, true);
#ifdef SAVE_DATA  
    save_tensor(SwiGLU_input ,"swiglu_input.bin", count);
#else
#endif
    // 2.swiGLU
    launchAct(SwiGLU_input, down_proj_input);// down_proj_input maybe can reuse swiglu_input buf, will validate it later
    DeviceSyncAndCheckCudaError();
#ifdef SAVE_DATA
    save_tensor(down_proj_input ,"down_proj_input.bin", count); 
#else
#endif
    // 3.down proj
    launchLinearGemm(down_proj_input, weights.down, ffn_output->as<T>(), cublas_wrapper, false, true);
    DeviceSyncAndCheckCudaError();
    this->freeBuf();
};

template class LLaMAFFNLayer<float>;
template class LLaMAFFNLayer<half>;
