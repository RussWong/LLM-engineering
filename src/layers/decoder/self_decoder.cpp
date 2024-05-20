#include <iostream>
#include "src/utils/macro.h"
#include "src/layers/decoder/self_decoder.h"
//(RussWong) note: layers文件夹下，很多操作后面我都加了`DeviceSyncAndCheckCudaError();`，大家可手动删除或者按照lesson30所示添加条件编译代码
template<typename T>
void LlamaSelfDecoder<T>::allocForForward(LLaMAAttentionDynParams& params)
{
    DataType type = getTensorType<T>(); 
    int batch_size = params.batch_size;
    decoder_residual = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units});
    decoder_residual->data = allocator->Malloc(decoder_residual->data, sizeof(T) * batch_size * hidden_units, false);

}
template<typename T>
void LlamaSelfDecoder<T>::freeBuf()
{
    allocator->Free(decoder_residual->data);
}
template<typename T>
void LlamaSelfDecoder<T>::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight<T>*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{
    allocForForward(dyn_params);
    Tensor* decoder_input = input_tensors["decoder_input"];
    Tensor* step = input_tensors["step"];
    Tensor* finished = input_tensors["finished"];
    Tensor* decoder_output = output_tensors["decoder_output"];
    Tensor* all_k_cache = output_tensors["all_k_cache"];
    Tensor* all_v_cache = output_tensors["all_v_cache"];
    Tensor* layer_id = input_tensors["layer_id"];
    DataType type_int = getTensorType<int>();
    LLM_CHECK_WITH_INFO(decoder_input->as<T>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(step->as<int>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");
    LLM_CHECK_WITH_INFO(finished->as<bool>()->data != nullptr, "the data ptr of tensor inserted into TensorMap is nullptr!");

    TensorMap self_attn_inputs{
        {"attention_input", decoder_input},
        {"layer_id", layer_id},
        {"step", step},
        {"finished", finished}
    };
    TensorMap self_attn_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    }; 
       
    for(int layer_id = 0; layer_id < num_layer; layer_id++) {
	//std::cout << "=============in layer " << layer_id << "==============" << "\n";
	if (layer_id > 0){
            TensorWrapper<int>* layer = new TensorWrapper<int>(Device::CPU, type_int, {1}, &layer_id);
            self_attn_inputs.insert("layer_id", layer);
        }
        decoder_input = self_attn_inputs["attention_input"];
        launchRMSNorm(decoder_input->as<T>(), //in&out, [bs, q_hidden_units]
                    decoder_residual, // = rmsnorm input hidden states, as input of next add residual
                    layerWeights[layer_id]->attn_norm_weight,//rmsnorm weights, [q_hidden_units]
                    rmsnorm_eps);
        DeviceSyncAndCheckCudaError();  
        selfAttn->forward(self_attn_inputs, self_attn_outputs, layerWeights[layer_id]->self_attn_weight, dyn_params);
        launchFusedAddBiasResidualRMSNorm(decoder_residual, //in residual from tensor before rmsnorm and return decoder_residual + decoder_output, [bs, q hidden_units]
                                          decoder_output->as<T>(), //in&out from attention output, [bs, q hidden_units]
                                          layerWeights[layer_id]->self_attn_weight.output, //bias
                                          layerWeights[layer_id]->ffn_norm_weight.gamma,//rmsnorm weights, [q hidden_units]
                                          rmsnorm_eps);
        DeviceSyncAndCheckCudaError();
        TensorMap ffn_inputs{
            {"ffn_input", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_output", decoder_output}
        };
        ffn->forward(ffn_inputs, ffn_outputs, layerWeights[layer_id]->ffn_weight, dyn_params);
        launchAddResidual(decoder_residual, //in, [bs, hidden_units]
                        decoder_output->as<T>(), //in&out, [bs, hidden_units]
                        true);
        
	DeviceSyncAndCheckCudaError();
        self_attn_inputs.insert("attention_input", decoder_output); // for next iter
    }
    // no intermedia buffer to free, so ignore call free
}

template class LlamaSelfDecoder<float>;
template class LlamaSelfDecoder<half>;
