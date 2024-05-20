#include <float.h> //FLT_MIN
#include <iostream>
#include "src/kernels/topK.h"
#include <cub/cub.cuh>

// Note: a b两个topK reduce输出一个topK
template <typename T, int K>
__device__ topK<T, K> reduce_functor(const topK<T, K> &a, const topK<T, K> &b)
{
	topK<T, K> res = a;
	for (int i = 0; i < K; i++) {
		res.insertHeap(b.val[i], b.id[i]);
	}
	return res;
}
// gridsize:bs * beam_width * BlockPerBeam
// blocksize:256
// shape infer: [bs, beam_width, vocab size] => [bs, beam_width, BlockPerBeam, K],在vocabsize的大小里选出blockPerBeam个topK
template <typename T, int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round1(const T *probs, const int vocab_size,
                                   int *topK_ids, T *topK_vals)
{
	typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
        __shared__ typename blockreduce::TempStorage tmp_storage;

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int row_id = bid / BlockPerBeam;
	int block_lane = bid % BlockPerBeam;
	topK<T, K> thread_topK;
	thread_topK.init();
	//thread local reduce
	for (int data_id = tid + block_lane * blockSize; data_id < vocab_size; data_id += BlockPerBeam * blockSize) {
		int data_offset = data_id + row_id * vocab_size;
		T data = probs[data_offset];
		thread_topK.insertHeap(data, data_offset);
        	// if (bid == 1 && data_id < 10) {
            // 		printf("ROUND1, 1st block, top1 vals = %f, top1 id = %d\n", data, data_offset);
        	// }
	}
//	typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
//	__shared__ typename blockreduce::TempStorage tmp_storage;
	topK<T, K> block_topk = blockreduce(tmp_storage).Reduce(thread_topK, reduce_functor<T, K>);

	if (tid == 0) {
		for (int k_offset = 0; k_offset < K; k_offset++) {
			int dst_offset = row_id * BlockPerBeam * K + block_lane * K + k_offset;
			topK_vals[dst_offset] = block_topk.val[k_offset];
			topK_ids[dst_offset] = block_topk.id[k_offset];
		}
	}

}
// shape infer: [bs, beam_width, BlockPerBeam, K] => [bs, beam_width, K] ，这是sampling的topK（=>[bs, beam_width, K]才是beamsearch topK），后期注意重写一个beamsearch的topK
// gridSize = bs
// blockSize = 256
template <typename T, int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round2(const int *topK_ids, const T *topK_vals,
                                   int *final_topK_ids, T *final_topK_vals)
{
        typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
        __shared__ typename blockreduce::TempStorage tmp_storage;

	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int row_id = bid;
	topK<T, K> thread_topK;
	thread_topK.init();
	//thread local reduce
	for (int data_id = tid; data_id < BlockPerBeam * K; data_id += blockSize) {
		int data_offset = data_id + bid * BlockPerBeam * K;
	
		thread_topK.insertHeap(topK_vals[data_offset], topK_ids[data_offset]);
        	// if (bid == 0 && data_id == 0) {
            // 		printf("ROUND2, 1st block, top1 vals = %f, top1 id = %d\n", topK_vals[data_offset], topK_ids[data_offset]);
        	// }
	}

//	typedef cub::BlockReduce<topK<T, K>, blockSize> blockreduce;
//	__shared__ typename blockreduce::TempStorage tmp_storage;
	topK<T, K> block_topk = blockreduce(tmp_storage).Reduce(thread_topK, reduce_functor<T, K>);

	if (tid == 0) {
		//int beam_id = (blockDim.x * blockIdx.x + tid) / BlockPerBeam / K;
		for (int k_offset = 0; k_offset < K; k_offset++) {
			int dst_offset = bid * K + k_offset;
			final_topK_vals[dst_offset] = block_topk.val[k_offset];
			final_topK_ids[dst_offset] = block_topk.id[k_offset];
		}
	}
}

template <typename T>
void launchTopKforBeamSearch(TensorWrapper<T> *probs,
                             // TensorWrapper<T>* topk_workspace
                             TensorWrapper<int> *tmp_topk_ids,
                             TensorWrapper<T> *tmp_topk_vals,
                             TensorWrapper<int> *final_topk_ids,
                             TensorWrapper<T> *final_topk_vals)
{
    int batch_size = probs->shape[0];
    int vocab_size = probs->shape[1];
    constexpr int BlockPerBeam = 8;
    constexpr int beam_width = 1;
    constexpr int K = 5;
    // buffer size
    // int topK_val_buf_size = batch_size * beam_width * BlockPerBeam * beam_width;
    // int topK_ids_buf_size = batch_size * beam_width * BlockPerBeam * beam_width;
    // int final_topK_val_buf_size = batch_size * beam_width; // sampling topK buf size, beamsearch topK size = [batch_size * beam_width * beam_width]
    // memory plan
    T *topK_vals = tmp_topk_vals->data;         // topK_val_buf_size
    int *topK_ids = tmp_topk_ids->data;         // topK_ids_buf_size
    T *final_topK_vals = final_topk_vals->data; // final_topK_val_buf_size
    int *final_topK_ids = final_topk_ids->data; // final_topK_val_buf_size
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxBlockNums = deviceProp.maxGridSize[0];
    int BlockNums1 = std::min(batch_size * beam_width * BlockPerBeam, maxBlockNums);
    int BlockNums2 = std::min(batch_size * beam_width, maxBlockNums);
    dim3 grid_round1(BlockNums1);
    dim3 block_round1(256);
    dim3 grid_round2(BlockNums2);
    dim3 block_round2(256);
    // debug info, better to retain: std::cout << "in cu file, before launch" << std::endl;
    topK_kernel_round1<T, K, 256, BlockPerBeam>
        <<<grid_round1, block_round1>>>(probs->data, vocab_size, topK_ids, topK_vals);
    topK_kernel_round2<T, K, 256, BlockPerBeam>
        <<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
    // debug info, better to retain: std::cout << "in cu file, after launch" << std::endl;
}

template void launchTopKforBeamSearch(TensorWrapper<float> *probs,
                                      TensorWrapper<int> *tmp_topk_ids,
                                      TensorWrapper<float> *tmp_topk_vals,
                                      TensorWrapper<int> *final_topk_ids,
                                      TensorWrapper<float> *final_topk_vals);
template void launchTopKforBeamSearch(TensorWrapper<half> *probs,
                                      TensorWrapper<int> *tmp_topk_ids,
                                      TensorWrapper<half> *tmp_topk_vals,
                                      TensorWrapper<int> *final_topk_ids,
                                      TensorWrapper<half> *final_topk_vals);
