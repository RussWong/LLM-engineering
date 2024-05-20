cmake_minimum_required(VERSION 3.8)

add_library(LlamaCtxAttn STATIC context_attention.cpp)
set_property(TARGET LlamaCtxAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaCtxAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaCtxAttn PUBLIC
#                             -lcudart
                             -lcudadevrt
                             qkv_bias_and_rope
                             concat_kv
#                             cublasWrapper
                             linear
                             fused_transpose_and_remv_pad
                             repeat_kv
                             mask_softmax)

add_library(LlamaselfAttn STATIC masked_self_attention.cpp)

set_property(TARGET LlamaselfAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaselfAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaselfAttn PUBLIC
                             -lcudart
                             -lcudadevrt
                             fused_decoder_self_attention
                             qkv_bias_and_rope
)
