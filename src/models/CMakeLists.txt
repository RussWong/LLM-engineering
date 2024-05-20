cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

add_library(Llama STATIC llama/llama.cpp)
set_property(TARGET Llama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llama PUBLIC LlamaCtxdecoder Llamaselfdecoder weightutils Llamaffn sample embeddingFunctor)
