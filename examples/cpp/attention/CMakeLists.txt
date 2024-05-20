cmake_minimum_required(VERSION 3.8)

add_executable(CtxAttnExample context_attn_example.cpp)
set_property(TARGET CtxAttnExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET CtxAttnExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(CtxAttnExample PUBLIC
                             -lcudart
                             -lcudadevrt
                             LlamaCtxAttn)
add_executable(selfAttnExample self_attn_example.cpp)
set_property(TARGET selfAttnExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET selfAttnExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(selfAttnExample PUBLIC
                            -lcudart
                            -lcudadevrt                             
                             LlamaselfAttn
			     linear)
