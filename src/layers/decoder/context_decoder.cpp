#include <iostream>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
#include "src/layers/decoder/context_decoder.h"
// (RussWong)note: in llama, all linear dont have bias
// (RussWong) note: layers文件夹下，很多操作后面我都加了`DeviceSyncAndCheckCudaError();`，大家可手动删除或者按照lesson30所示添加条件编译代码
template<typename T>
void LlamaContextDecoder<T>::allocForForward(LLaMAAttentionDynParams& params)
{
    int num_tokens = params.num_tokens;
    int batch_size = params.batch_size;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    DataType type_int = getTensorType<int>(); 
    decoder_residual = new TensorWrapper<T>(Device::GPU, type, {num_tokens, hidden_units});
    attention_mask = new TensorWrapper<T>(Device::GPU, type, {batch_size, max_q_len, max_k_len});
    padding_offset = new TensorWrapper<int>(Device::GPU, type_int, {batch_size, max_q_len});
    cum_seqlens = new TensorWrapper<int>(Device::GPU, type_int, {batch_size + 1});
    decoder_residual->data = allocator->Malloc(decoder_residual->data, sizeof(T) * num_tokens * hidden_units, false);
    attention_mask->data = allocator->Malloc(attention_mask->data, sizeof(T) * batch_size * max_q_len * max_k_len, false);
    padding_offset->data = allocator->Malloc(padding_offset->data, sizeof(int) * batch_size * max_q_len, false);
    cum_seqlens->data = allocator->Malloc(cum_seqlens->data, sizeof(int) * (batch_size + 1), false);
}
template<typename T>
void LlamaContextDecoder<T>::freeBuf()
{
    allocator->Free(attention_mask->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(padding_offset->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(cum_seqlens->data);
    DeviceSyncAndCheckCudaError();
}
template<typename T>
void LlamaContextDecoder<T>::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight<T>*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{
    allocForForward(dyn_params);
    Tensor* seq_lens = input_tensors["input_length"];
    // 1. calculate padding offset
    // shape:
        //seq_lengths:[batch size]
        //output cum_seqlens:[batch size + 1], first ele is 0
        //output padding_offset:[batch size * max q len]
    launchCalPaddingoffset(padding_offset, //out
                           cum_seqlens, //out
                           seq_lens->as<int>()); // in
    DeviceSyncAndCheckCudaError();
    //2. build causal mask
    Tensor* context_length = input_tensors["context_length"];
    launchBuildCausalMasks<T>(attention_mask, //out
                            seq_lens->as<int>(), //q, input lens, [bs]
                            context_length->as<int>());//k, context lens, [bs]
    DeviceSyncAndCheckCudaError();
    // 3. context attn
    Tensor* history_length = input_tensors["history_length"];
    Tensor* decoder_output = output_tensors["decoder_output"];
    Tensor* all_k_cache = output_tensors["all_k_cache"];
    Tensor* all_v_cache = output_tensors["all_v_cache"];
    DataType type_int = getTensorType<int>();
    DataType type = getTensorType<T>();
    Tensor* layer_id = input_tensors["layer_id"];
    Tensor* decoder_input = input_tensors["decoder_input"];
    LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(history_length->as<int>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    TensorMap ctx_attn_inputs{
        {"attention_input", decoder_input},
        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", seq_lens},
        {"context_length", context_length},
        {"attention_mask", attention_mask},
        {"layer_id", layer_id}
    };
    TensorMap ctx_attn_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    // same buffer between layers, reuse
    for(int layer_id = 0; layer_id < num_layer; layer_id++) {//num_layer; layer_id++) {
        //std::cout << "==============in layer " << layer_id << "==============" << "\n"; 
        if (layer_id > 0){
            TensorWrapper<int>* layer = new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id);
            ctx_attn_inputs.insert("layer_id", layer);
        }

        decoder_input = ctx_attn_inputs["attention_input"];
        launchRMSNorm(decoder_input->as<T>(), //in&out, [num tokens, q_hidden_units]
                    decoder_residual, // = rmsnorm input hidden states, used for next add residual
                    layerWeights[layer_id]->attn_norm_weight,//rmsnorm weights, [q_hidden_units]
                    rmsnorm_eps);
        DeviceSyncAndCheckCudaError();  
        ctxAttn->forward(ctx_attn_inputs, ctx_attn_outputs, layerWeights[layer_id]->self_attn_weight, dyn_params, ctxAttn->GetAttnStaticParams());
        // RussWong note:这里的residual是上一个rmsnorm的输入
        launchFusedAddBiasResidualRMSNorm(decoder_residual, //in residual from tensor before rmsnorm and return decoder_residual + decoder_output, [num tokens, hidden_units]
                                        decoder_output->as<T>(), //in&out from attention output, [num tokens, hidden_units]
                                        layerWeights[layer_id]->self_attn_weight.output, //bias
                                        layerWeights[layer_id]->ffn_norm_weight.gamma,//rmsnorm weights, [hidden_units]
                                        rmsnorm_eps);
        DeviceSyncAndCheckCudaError();
        #ifdef SAVE_DATA
            save_tensor(decoder_output->as<T>() ,"ffn_input.bin", layer_id);
        #else
        #endif
        TensorMap ffn_inputs{
            {"ffn_input", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_output", decoder_output}
        };
	dyn_params.is_ctx = true; // used to distinguish ffn is in context decoder or self decoder, so as to reduce print infos
        ffn->forward(ffn_inputs, ffn_outputs, layerWeights[layer_id]->ffn_weight, dyn_params);
        #ifdef SAVE_DATA
            save_tensor(decoder_output->as<T>() ,"ffn_output.bin", layer_id);
        #else
        #endif        
        // RussWong note:这里的residual为上一个launchFusedAddBiasResidualRMSNorm中加了redisual后的hidden states
        launchAddResidual(decoder_residual, //residual, [num tokens, hidden_units]
                        decoder_output->as<T>() //in&out, [num tokens, hidden_units]
                        );
        DeviceSyncAndCheckCudaError();
        ctx_attn_inputs.insert("attention_input", decoder_output);
    }
    freeBuf();
    DeviceSyncAndCheckCudaError();
}

template class LlamaContextDecoder<float>;
template class LlamaContextDecoder<half>;
