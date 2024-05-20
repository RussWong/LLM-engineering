cmake_minimum_required(VERSION 3.8)

add_executable(ctxDecoderExample context_decoder_example.cpp)
set_property(TARGET ctxDecoderExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ctxDecoderExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ctxDecoderExample PUBLIC
                             embeddingFunctor
                             LlamaCtxdecoder)

add_executable(selfDecoderExample self_decoder_example.cpp)
set_property(TARGET selfDecoderExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET selfDecoderExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(selfDecoderExample PUBLIC
                            Llamaselfdecoder)