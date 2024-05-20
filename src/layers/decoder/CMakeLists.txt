cmake_minimum_required(VERSION 3.8)

add_library(LlamaCtxdecoder STATIC context_decoder.cpp)
set_property(TARGET LlamaCtxdecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaCtxdecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaCtxdecoder PUBLIC
                                        LlamaCtxAttn
                                        Llamaffn 
                                        llamaweights
                                        cal_paddingoffset
                                        build_casual_mask
                                        rmsnorm
                                        fused_addresidual_norm
                                        add_residual
)

add_library(Llamaselfdecoder STATIC self_decoder.cpp)
set_property(TARGET Llamaselfdecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llamaselfdecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llamaselfdecoder PUBLIC
                                        LlamaselfAttn
                                        Llamaffn 
                                        llamaweights
                                        rmsnorm
                                        fused_addresidual_norm
                                        add_residual
)
