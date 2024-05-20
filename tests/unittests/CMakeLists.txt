add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)

add_executable(rms_norm
    test_rmsnorm.cu
)
target_link_libraries(
    rms_norm PUBLIC
    -lcudart
    -lcudadevrt
    rmsnorm)

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(
  paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    cal_paddingoffset) 

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
   causalmask PUBLIC
    -lcudart
    -lcudadevrt
    build_casual_mask) 

add_executable(testlinear
    test_linear.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    testlinear PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

add_executable(debug
    test_data_compare.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    debug PUBLIC
    -lcudart
    -lcudadevrt) 

add_executable(bmm
    test_bmm.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
    bmm PUBLIC
    -lcudart
    -lcudadevrt
    linear) 

add_executable(biasRope
    test_bias_and_RoPE.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
  biasRope PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope) 

add_executable(test_concat_kv
    test_concat_kv.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
    test_concat_kv PUBLIC
    -lcudart
    -lcudadevrt
    concat_kv) 

add_executable(test_repeat_kv
    test_repeat_kv.cu
)
target_link_libraries( 
    test_repeat_kv PUBLIC
    -lcudart
    -lcudadevrt
    repeat_kv) 

add_executable(test_mask_softmax
    test_mask_softmax.cu
)
target_link_libraries(  
    test_mask_softmax PUBLIC
    -lcudart
    -lcudadevrt
    mask_softmax) 

add_executable(test_fused_trans_remv_pad
    test_fused_trans_remv_pad.cu
)
target_link_libraries(  
    test_fused_trans_remv_pad PUBLIC
    -lcudart
    -lcudadevrt
    fused_transpose_and_remv_pad) 

add_executable(test_fused_addresidual_norm
    test_fused_addresidual_norm.cu
)
target_link_libraries(  
    test_fused_addresidual_norm PUBLIC
    -lcudart
    -lcudadevrt
    fused_addresidual_norm) 

add_executable(test_act
    test_act.cu
)
target_link_libraries(  
    test_act PUBLIC
    -lcudart
    -lcudadevrt
    act) 

add_executable(test_topk
    test_topk.cu
)
target_link_libraries(  
    test_topk PUBLIC
    -lcudart
    -lcudadevrt
    topk) 

add_executable(test_fused_decoder_attention
    test_fused_decoder_attention.cu
)
target_link_libraries(  
    test_fused_decoder_attention PUBLIC
    -lcudart
    -lcudadevrt
    fused_decoder_self_attention)

add_executable(test_sampling
    test_sampling.cu
)
target_link_libraries(  
    test_sampling PUBLIC
    -lcudart
    -lcudadevrt
    sampling) 

add_executable(test_residual
    test_residual.cu
)
target_link_libraries(  
    test_residual PUBLIC
    -lcudart
    -lcudadevrt
    add_residual) 
