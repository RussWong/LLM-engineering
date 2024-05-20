#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/weights/base_weights.h"
#include "src/weights/llama/norm_weights.h"
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"
template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units]
                                    TensorWrapper<T>* residual, 
                                    TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<T>& norm,
                                    T* scale, //RMSNorm weights
                                    float eps);
