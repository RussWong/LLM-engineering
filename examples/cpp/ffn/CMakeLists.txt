cmake_minimum_required(VERSION 3.8)

add_executable(ffnExample ffn_example.cpp)
set_property(TARGET ffnExample PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ffnExample PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ffnExample PUBLIC Llamaffn)