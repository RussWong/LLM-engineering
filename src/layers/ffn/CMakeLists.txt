cmake_minimum_required(VERSION 3.8)

add_library(Llamaffn STATIC ffn.cpp)
set_property(TARGET Llamaffn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET Llamaffn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(Llamaffn PUBLIC
                             -lcudart
                             -lcudadevrt
                             act
                             linear)