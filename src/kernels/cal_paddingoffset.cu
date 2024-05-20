#include "src/kernels/cal_paddingoffset.h"
// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
// TODO: we first use serial algo, then can enhance to CUDA scan algo

__global__ void CalPaddingoffset(int*         padding_offset, 
                                int*         cum_seqlens,
                                const int*   input_lengths, //actual input lens
                                const int    batch_size,
                                const int    max_q_len) {
    int ind = 0;
    int cum_offset = 0;
    int total_seqlen = 0;
    for(int b = 0; b < batch_size; b++) {
        int seqlen = input_lengths[b];

        cum_seqlens[b] = total_seqlen;
        // each token in one seq has same cum offset
        for (int i = 0; i < seqlen; i++) {
            padding_offset[ind] = cum_offset;
            ind++;
        }
        cum_offset += max_q_len - seqlen;
        total_seqlen += seqlen;
    }
    cum_seqlens[batch_size] = total_seqlen;
}

void launchCalPaddingoffset(TensorWrapper<int>* padding_offset, 
                            TensorWrapper<int>* cum_seqlens,
                            TensorWrapper<int>* input_lengths)//actual input lens
{
    const int batch_size = padding_offset->shape[0];                            
    const int max_q_len = padding_offset->shape[1]; 
    LLM_CHECK_WITH_INFO(batch_size == input_lengths->shape[0], "input lenghts numbers should equal to padding offset bs dim!") ;                        
    LLM_CHECK_WITH_INFO(batch_size == cum_seqlens->shape[0] - 1, "cum seqlen numbers should equal to padding offset bs dim + 1!") ;                        
    CalPaddingoffset<<<1, 1>>>( 
        padding_offset->data, cum_seqlens->data, input_lengths->data, batch_size, max_q_len
    );
}