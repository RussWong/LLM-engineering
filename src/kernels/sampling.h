#pragma once
#include <curand.h>
#include <curand_kernel.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "src/utils/tensor.h"
#include "src/utils/params.h"

template<typename T>
void launchSampling(TensorWrapper<int>* topk_id,
                    TensorWrapper<T>* topk_val,
                    TensorWrapper<int>* seqlen,
                    TensorWrapper<bool>* is_finished,
                    TensorWrapper<int>* output_id,
                    IntDict& params);