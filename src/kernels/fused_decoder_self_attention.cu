#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/utils/cuda_debug_utils.cuh"
#include "src/kernels/fused_decoder_self_attention.h"
// kv cache shape = [numlayers, bs, kv head num, max_seq_len, head size]
// bug1: scale's dtype must be float ,not int
// bug2: mha_kernel_params struct's pointer is on CPU, not GPU, which cause we dont run the cuda kernel, so add cudacheck is a must!
// bug3: blockreduce res should use tid=0 to write into smem
// bug4: GQA, kv_head_num brd to head_num, we can automaticly do this by head id index like lmdeploy
// half or float version: the logits and mha output both are fp32 type, q k v are all accessed vectorizedly
template<typename T>
__device__ T warpReduceSum(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;

}
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpsum[64];//why add static?or will report incomplete type
    // returned val is the sum computed by 0th thread.
    val = warpReduceSum<T>(val);
    //note: here return val of warpreducesum should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpsum[tid] : (T)0.0f;
    return warpReduceSum<T>(warp_val);

}
template<typename T>
__device__ T warpReduceMax(T val){
    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpmax[64];
    // returned val is the max computed by 0th thread.
    val = warpReduceMax(val); // remove <T> can ignore the multi-overloaded error?
    //note: here return val of warpreducemax should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    T warp_val = tid < warp_nums ? warpmax[tid] : (T)0;
    return warpReduceMax(warp_val);
}

inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    // (RussWong) note: 每个token所属的id, 它的freq值都是固定的, id的上限为max position embedding
    // t_step表示token id（这里考虑了多轮对话历史上下文长度)
    // 每个freq值对应于zid = head size维度上0 2 4 6 ... 64带入下式计算
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim); //rot_embed_dim = 128
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 GetRoPEres(float data, float data_rotate, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * data - coef.y * data_rotate;
    rot_v.y = coef.x * data_rotate + coef.y * data;
    return rot_v;
}
// for chatglm2 rope
// inline __device__ float2 GetRoPEres(const float2 v, const float2 coef)
// {
//     float2 rot_v;
//     rot_v.x = coef.x * v.x - coef.y * v.y;
//     rot_v.y = coef.x * v.y + coef.y * v.x;
//     return rot_v;
// }

// inline __device__ half2 GetRoPEres(const half2 v, const float2 coef)
// {
//     float2 fv     = __half22float2(v);
//     float2 rot_fv = GetRoPEres(fv, coef);
//     return __float22half2_rn(rot_fv);
// }

// inline __device__ void apply_RoPE(half2& q, half2& k, int tid, int rot_embed_dim, float base, float t_step)
// {
//     if (2 * tid >= rot_embed_dim) {
//         return;
//     }
//     const auto coef = GetRoPEfreq(2 * tid, rot_embed_dim, base, t_step);
//     q               = GetRoPEres(q, coef);
//     k               = GetRoPEres(k, coef);
// }

// inline __device__ void apply_RoPE(float4& q, float4& k, int tid, int rot_embed_dim, float base, float t_step){
//     if(4 * tid >= rot_embed_dim){
//         return;
//     }
//     TwoFloat2& q_ = *reinterpret_cast<TwoFloat2*>(&q);
//     TwoFloat2& k_ = *reinterpret_cast<TwoFloat2*>(&k);
    
//     float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
//     q_.x = GetRoPEres(q_.x ,coef0);
//     float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
//     q_.y = GetRoPEres(q_.y ,coef1);
//     k_.x = GetRoPEres(k_.x ,coef0);
//     k_.y = GetRoPEres(k_.y ,coef1);
// }

// block and thread allocation
// 1 block -> head size，后续可改进为1 warp -> 1 head size or 1 block -> multi head size
// 1 grid -> bs * num heads
// q; input vec [bs, q num heads, 1, head size]
// k; input vec [bs, kv num heads, 1, head size]
// v; input vec [bs, num heads, 1, head size]
// k_cache; output,[bs, max_seq_len, kv num heads, head size] from prompt phase
// v_cache; output,[bs, max_seq_len, num heads, head size] from prompt phase
template<typename T>
__global__ void masked_MHA_kernel(T* q,
                    T* k,
                    T* v,
                    T* qkv_bias,
                    T* k_cache,
                    T* v_cache,
                    T* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){// rsqrt(dh)
    int tid = threadIdx.x;
    int q_batch_id = blockIdx.x / head_num;
    int q_head_id = blockIdx.x % head_num;
    // (RussWong) note: below one line is wrong kv head id access way.
    //int kv_head_id = bid % kv_head_num;
    int kv_head_id = q_head_id / (head_num / kv_head_num);
    int kv_batch_id = q_batch_id;
    int batch_stride = head_num * head_size;
    int kv_batch_stride = kv_head_num * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;

    int vec_size = Vec<T>::size;
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
    int cache_offset = kv_batch_id * kv_head_num * max_seq_len * head_size +
                        kv_head_id * max_seq_len * head_size + tid * vec_size;
    int step_stride = head_size;
    float scale = rsqrt(float(head_size));

    using Vec_t = typename Vec<T>::Type;
    Vec_t qvec, kvec, vvec;
    const T* q_mem = q;
    const T* k_mem = k;
    const T* v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&q_mem[q_offset_vec]));
        // (RussWong) note: will enable below code lines when qkv bias is available
        // if (qkv_bias != nullptr){
	    //     Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]);
        //     for(int i = 0; i < vec_size; i++) {
        //         reinterpret_cast<float*>(&qvec)[i] += reinterpret_cast<float*>(&q_bias)[i];
        //     }
	    // }
        kvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&k_mem[k_offset_vec]));        
        vvec = *reinterpret_cast<Vec_t*>(const_cast<T*>(&v_mem[k_offset_vec]));
        // (RussWong) note: will enable below code lines when qkv bias is available
        // if (qkv_bias != nullptr){
	    //     Vec_t v_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]);
        //     for(int i = 0; i < vec_size; i++) {
        //         reinterpret_cast<float*>(&vvec)[i] += reinterpret_cast<float*>(&v_bias)[i];
        //     }
	    // }
    }
    // q k smem for block reduce
    // (RussWong) note: define dynamic smem type is char type!! not T
    // mainly to show how to memory plan dynamic smem 
    extern __shared__ char sqk[];
    T* sq_scalar = reinterpret_cast<T*>(sqk); // q存在smem的必要性:在第step行把q存进smem，之前的step-1行可以直接从smem load to reg参与计算
    float* logits = reinterpret_cast<float*>(sq_scalar + head_size); // logits存在smem的必要性:所有线程reduce的结果存到logits，需要smem
    Vec_t* sq = reinterpret_cast<Vec_t*>(sq_scalar);
    if (tid * vec_size < head_size) {
        sq[tid] = qvec;
    }
    __syncthreads();
    // (RussWong) note: convert some constant scalar to vector type for vectorizedly computation
    float zero = 0.0f;
    Vec_t zero_f4 = scalar_cast_vec<Vec_t, T>(zero);
    float4 scale_f4 = scalar_cast_vec<float4, float>(scale);

    for(int iter = 0; iter < step; iter++) {
        // (RussWong) note: every iter,  q and k vector's shape = [1, head size]
        // reuse k cache
        // TODO: 或许可以在每个step省略掉前step-1的qk dot
        Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) : zero_f4;
        // (RussWong) note: when final step, update k cache and fetch k from input k vec, rather than kv cache
        if (iter == step - 1 && tid * vec_size < head_size) {
            // TODO: update k cache with k with bias add when model has qkv bias
            *reinterpret_cast<Vec_t*>(&k_cache[iter * step_stride + cache_offset]) = kvec;
            kvec_qk = kvec;
        }
        Vec_t qk = zero_f4;
        qk.x = (tid * vec_size < head_size) ? sq[tid].x * kvec_qk.x * scale_f4.x : zero;
        qk.y = (tid * vec_size < head_size) ? sq[tid].y * kvec_qk.y * scale_f4.y : zero;
        qk.z = (tid * vec_size < head_size) ? sq[tid].z * kvec_qk.z * scale_f4.z : zero;
        qk.w = (tid * vec_size < head_size) ? sq[tid].w * kvec_qk.w * scale_f4.w : zero;
        T qk_acc = qk.x + qk.y + qk.z + qk.w;
        //block reduce using multi warp reduce
        //TODO: maybe broadcast the attn score to each thread of the block in blockreducesum
        T attn_score = blockReduceSum<T>(qk_acc);
        if(tid == 0) {
            logits[iter] = attn_score;
	}
        __syncthreads();
    }
    //softmax(logits), logits.shape = [bs, num heads, 1, step]
    T local_logits = tid < step ? (T)logits[tid] : 0;
    __shared__ float row_max, fenmu;
    
    T block_max = blockReduceMax<T>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    __syncthreads();
    T fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    
    T block_fenmu = blockReduceSum<T>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu + 1e-6;
    }
    __syncthreads();
    if(tid < step) {
        logits[tid] = (T)(fenzi / fenmu);
    }
    __syncthreads();

    // logits*V = [bs, num heads, 1, step] * [bs, kv num heads, step, head size]
    if (tid * vec_size < head_size) {
        // note: above if condition is < head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
        // so here we use acc O to acc the one ele logits * one ele v every step iter
        // same computation logic with CUDA lesson52 and lesson53
        Vec_t O = scalar_cast_vec<Vec_t, T>(0.0f);
        for(int iter = 0; iter < step; iter++) {
            Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]);
            // T value = v_cache[ite * cache_offset + k_offset];
            // when final step, update k cache
	    if (iter == step - 1) {
                // TODO: update k cache with k with bias add
                *reinterpret_cast<Vec_t*>(&v_cache[iter * step_stride + cache_offset]) = vvec;
                // kv cache does not cache cur step kv, so fetch from input v vec of cur step
                vvec_qkv = vvec;
            }
            O.x += vvec_qkv.x * logits[iter];
            O.y += vvec_qkv.y * logits[iter];
            O.z += vvec_qkv.z * logits[iter];
            O.w += vvec_qkv.w * logits[iter];
        }
        *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = O;
    }
}

template<>
__global__ void masked_MHA_kernel(half* q,
                    half* k,
                    half* v,
                    half* qkv_bias,
                    half* k_cache,
                    half* v_cache,
                    half* mha_output,
                    const int batch_size,
                    const int head_num,
                    const int kv_head_num,
                    const int max_seq_len,
                    const int head_size,
                    const int step,
                    int   rotary_embedding_dim,
                    float rotary_embedding_base){// rsqrt(dh)
        // (RussWong) note: to sync with newest fp32 mha
//     int tid = threadIdx.x;
//     //int bid = blockIdx.x;
//     int q_head_id = blockIdx.x;
//     int q_batch_id = blockIdx.y;
//     int kv_head_id = q_head_id / head_num / kv_head_num;
//     int kv_batch_id = q_batch_id;
//     //int q_head_id = bid % head_num;
//     //int q_batch_id = bid / head_num;
//     //int kv_head_id = bid % kv_head_num;
//     //int kv_batch_id = bid / kv_head_num;

//     int batch_stride = head_num * head_size;
//     int kv_batch_stride = kv_head_num * head_size;
//     int head_stride = head_size;
//     int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
//     int k_offset = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid;
//     int cache_offset = batch_size * kv_batch_stride;

//     int vec_size = Vec<half>::size;
//     int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
//     int k_offset_vec = kv_batch_id * kv_batch_stride + kv_head_id * head_stride + tid * vec_size;
//     half scale = __float2half(rsqrt(float(head_size)));
//     using Vec_t = typename Vec<half>::Type;
//     Vec_t qvec, kvec, vvec;
//     Vec_t scale_vec = scalar_cast_vec<Vec_t>(scale);
//     //reuse q k v reg from rope
//     const half* q_mem = q;
//     const half* k_mem = k;
//     const half* v_mem = v;
//     if (tid * vec_size < head_size) {
//         qvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&q_mem[q_offset_vec]));
//         if (qkv_bias != nullptr){
//             Vec_t q_bias = *reinterpret_cast<Vec_t*>(&qkv_bias[q_head_id * head_size + tid * vec_size]);
//             qvec = __hadd2(qvec, q_bias);
//         }
//         kvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&k_mem[k_offset_vec]));
//         if (qkv_bias != nullptr){
//             Vec_t k_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size]);
//             kvec = __hadd2(kvec, k_bias);
//         }
//         //apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, step);
//         vvec = *reinterpret_cast<Vec_t*>(const_cast<half*>(&v_mem[k_offset_vec]));
//         if (qkv_bias != nullptr){
//             Vec_t v_bias =*reinterpret_cast<Vec_t*>(&qkv_bias[kv_head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]);
//             vvec = __hadd2(vvec, v_bias);
//         }
// 	apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, step);

//     }
//     // q k smem for block reduce
//     extern __shared__ char sqk[];
//     half* sq = reinterpret_cast<half*>(sqk);
//     // half* sk = sq + head_size;
//     // //float* logits = reinterpret_cast<float*>(sk + head_size);
//     // half* sv = sk + head_size;
//     float* logits = reinterpret_cast<float*>(sq + head_size);
//     //sq[tid] = q_mem[qkv_offset];

//     Vec_t* sq_vec = reinterpret_cast<Vec_t*>(sq);
//     // Vec_t* sk_vec = reinterpret_cast<Vec_t*>(sk);
//     // Vec_t* sv_vec = reinterpret_cast<Vec_t*>(sv);
//     if (tid * vec_size < head_size) {
//         // *reinterpret_cast<Vec_t*>(&sq[tid * vec_size]) = qvec;
//         sq_vec[tid] = qvec;
//     }
//     __syncthreads();
//     half zero = (half)0.0f;
//     Vec_t zero_h2 = scalar_cast_vec<Vec_t, half>(zero);
//     Vec_t scale_h2 = scalar_cast_vec<Vec_t, half>(scale);
//     // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
//     for(int iter = 0; iter < step; iter++) {
//         // every iter,  q and k's shape = [1, head size]
//         // reuse k cache
//         // float k = k_cache[iter * cache_offset + qkv_offset];
//         //或许可以在每个step省略掉前step-1的qk dot
//         // sk_vec[tid]= *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]);
//         // __syncthreads();
//         Vec_t kvec_qk = tid * vec_size < head_size ? *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) : zero_h2;
//         // when final step, update k cache
//         if (iter == step - 1 && tid * vec_size < head_size) {
//             // TODO: update k cache with k with bias add
//             //k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
//             //sk[tid] = k_mem[qkv_offset];
//             *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) = kvec;
//             kvec_qk = kvec;         
//         }

//         // sq[tid] = q_mem[qkv_offset];
//         __syncthreads();
//         Vec_t qk = (tid * vec_size < head_size) ? __hmul2(__hmul2(sq_vec[tid], kvec_qk), scale_h2) : zero_h2;
//         //block reduce using multi warp reduce
//         float qk_fp32 = __half2float(qk.x) + __half2float(qk.y);
//         float attn_score = blockReduceSum<float>(qk_fp32);
//         if(tid == 0) {
//             logits[iter] = attn_score;
// 	    //float q_tmp = (float)(sq_vec[0].x);
// 	    //float k_tmp = (float)(sk_vec[0].x);
// 	    //float scale_tmp = (float)(scale_vec.x);
//             //printf("iter = %d, step=%d, blockIdx.x = %d, in cuda, logits[%d]=%f, qk_fp32 = %f, q_tmp=%f, k_tmp=%f, scale_tmp=%f\n",iter, step, blockIdx.x, iter, logits[iter], qk_fp32, q_tmp, k_tmp, scale_tmp);
// 	}
//         __syncthreads();
//     }
//     //__syncthreads();
//     //softmax(logits), logits.shape = [bs, num heads, 1, step]
//     //if(tid < step){
//     	//printf("logits[%d]=%f\n", tid, logits[tid]);
//     //}
//     float local_logits = tid < step ? logits[tid] : 0;
//     __shared__ float row_max, fenmu;
    
//     float block_max = blockReduceMax<float>(local_logits);
//     if (tid == 0){
//         row_max = block_max;
//     }
//     __syncthreads();
//     float fenzi = tid < step ? expf(local_logits - row_max) : 0;
//     //if(tid < step) {
//     //	printf("after expf, row_max=%f, fenzi=%f, logits=%f\n", row_max, fenzi, local_logits);
//     //}
//     float block_fenmu = blockReduceSum<float>(fenzi);
//     if (tid == 0){
//         fenmu = block_fenmu;
//     }
//     __syncthreads();
//     if(tid < step) {
        
// 	logits[tid] = (float)(fenzi / fenmu);
// //	printf("in cuda, row_max=%f, fenzi=%f, fenmu=%f, logits=%f\n", row_max, fenzi, fenmu, logits[tid]);
	
//     }
//     __syncthreads();

//     // logits*V = [bs, num heads, 1, step] * [max_seq_len or step, bs, num heads, head size]
//     if (tid * vec_size < head_size) {
//         // note: here is head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
//         // so here we use acc O to acc the one ele logits * one ele v every step iter
//         float2 O = scalar_cast_vec<float2>(0.0f);
//         //O.x = 0.0f;
//         //O.y = 0.0f;
//         for(int iter = 0; iter < step; iter++) {
//             // sv_vec[tid]= *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]);
//             // __syncthreads();
//             Vec_t vvec_qkv = *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]);
//             // when final step, update k cache
//             if (iter == step - 1) {
//                 // TODO: update k cache with k with bias add
//                 // v_cache[iter * cache_offset + k_offset] = v_mem[k_offset];
//                 // sv[tid] = v_mem[k_offset];
//                 *reinterpret_cast<Vec_t*>(&v_cache[iter * cache_offset + k_offset_vec]) = vvec;
//                 vvec_qkv = vvec;  
//             }
// 	    __syncthreads();
//             //if(bid==0 && tid == 0){
//             //printf("when tid=0, v cache = %f\n", sv[tid]);
//             O.x += (logits[iter] * __half2float(vvec_qkv.x));
//             O.y += (logits[iter] * __half2float(vvec_qkv.y));
//             //O += sv[tid] * logits[iter];
//             __syncthreads();
//         }
        
//         // float* mha_output_fp32 = reinterpret_cast<float*>(mha_output);
//         *reinterpret_cast<Vec_t*>(&mha_output[q_offset_vec]) = __float22half2_rn(O);
//     }
}

template<typename T>
void launchDecoderMaskedMHA(TensorWrapper<T>* qkv_buf,
                            BaseWeight<T>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache,
                            TensorWrapper<T>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<T>* mha_output,
                            LLaMAAttentionStaticParams& static_params){
    // (RussWong)note: we should carefully get shape from tensorwrapper, DON'T be GOT wrong value
    // the shape should be aligned with the CUDA kerne we wrote
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2];
    const int max_seq_len = k_cache->shape[3]; 
    int head_num = qkv_head_num - 2 * kv_head_num;
    const int head_size = qkv_buf->shape[2];
    const int cur_step = step->getVal();
    const int layer = layer_id->getVal();
    const int layer_offset = layer * max_seq_len * batch_size * kv_head_num * head_size;
    size_t smem_size_bytes = head_size * sizeof(T) + cur_step * sizeof(float);
    T* qkv_data = qkv_buf->data;
    //qkv_data.shape = [bs, 1, qkv_head_num,head_size]
    T* q = qkv_data;
    T* k = qkv_data + head_num * head_size;
    T* v = qkv_data + (head_num + kv_head_num) * head_size;

    int   rotary_embedding_dim = static_params.rotary_embedding_dim;
    float rotary_embedding_base = static_params.rotary_embedding_base;
    int   max_position_embeddings = static_params.max_position_embeddings;
    bool  use_dynamic_ntk = static_params.use_dynamic_ntk;
    dim3 grid(head_num * batch_size);
    dim3 block(head_size); //vec size = 4 for fp32
    masked_MHA_kernel<T><<<grid, block, smem_size_bytes>>>(q,
                                                            k,
                                                            v,
                                                            /*(T*)*/qkv.bias,
                                                            k_cache->data + layer_offset,
                                                            v_cache->data + layer_offset,
                                                            mha_output->data,
                                                            batch_size,
                                                            head_num,
                                                            kv_head_num,
                                                            max_seq_len,
                                                            head_size,
                                                            cur_step,
                                                            rotary_embedding_dim,
                                                            rotary_embedding_base);
#ifdef PRINT_DATA
    print_data<<<1, 1>>>(mha_output->data, true);
#else
#endif
}

template void launchDecoderMaskedMHA(TensorWrapper<float>* qkv_buf,
                            BaseWeight<float>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<float>* k_cache,
                            TensorWrapper<float>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<float>* mha_output,
                            LLaMAAttentionStaticParams& static_params);

template void launchDecoderMaskedMHA(TensorWrapper<half>* qkv_buf,
                            BaseWeight<half>& qkv,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<half>* k_cache,
                            TensorWrapper<half>* v_cache,
                            TensorWrapper<bool>* finished,
                            TensorWrapper<int>* step,
                            TensorWrapper<half>* mha_output,
                            LLaMAAttentionStaticParams& static_params);
