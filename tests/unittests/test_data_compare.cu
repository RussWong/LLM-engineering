#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <stdio.h>
#include <fstream>
#include "src/utils/macro.h"
#include "src/utils/debug_utils.h"
// (RussWong)note:
// this test is for debug, to compare intermediate tensor and HF intermediate tensor
// and the intermediate tensor will be saved in file when you compile the proj by `cmake .. -DSAVE_DATA=ON && make -j8`
// before run, you should change the path to your local right dir
// `./debug` to compare 

std::vector<float> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    size_t dim0 = 1, dim1 = 1;
    if (shape.size() > 2) {
        dim0 = shape[0] * shape[1];
        dim1 = shape[2];
    }

    if (shape.size() == 2) {
        dim0 = shape[0];
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        std::cout << "shape is zero, skip loading weight from file: " << filename << std::endl;
        return std::vector<float>();
    }

    std::vector<float> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "file" << filename << "cannot be opened, loading model fails!" << std::endl;
        return std::vector<float>();
    }

    size_t loaded_data_size = sizeof(float) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    std::cout << "Read " << std::to_string(loaded_data_size) << " bytes from " << filename << std::endl;
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<float>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}
void internalFunc(float* ptr, std::vector<size_t> shape, std::string filename) {
    std::vector<float> host_array = loadWeightFromBinHelper(shape, filename);
    if (host_array.empty()) {
        std::cout << "[warning] data from file is empty!!" << "\n";
        return;
    }
    // copy host_array to our defined ptr
    memcpy(ptr, host_array.data(), host_array.size());
    return;
}
void loadWeights(float* ptr1, std::string weight_path, int shape0, int shape1) // weighttype参数比较多余
{
    // load attn output
    internalFunc(ptr1, {(size_t)shape0, (size_t)shape1}, weight_path);

}
void loadWeights_trans(float* ptr1, std::string weight_path, int shape0, int shape1) // weighttype参数比较多余
{
    // load attn output
    internalFunc(ptr1, {(size_t)shape0, (size_t)shape1}, weight_path);

}

bool CheckResult(float* CPUoutput, float* GPUoutput, int in_size) {
    for(int i = 0; i < in_size; i++) {
	if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
	    printf("the %dth res is wrong, onellm = %f, trans = %f\n", i, CPUoutput[i], GPUoutput[i]);
    	}
    }
    return true;
}
// 1.for example: the path of two data files is below, and you should replace L101&L102 with the two
// /home/data/trans/q_buf_after_rope_trans.bin
// /home/data/onellm/q_buf_after_rope.bin
// 2.And you should change the L93&L94 to the right data size according to your tensor shape of the data file
int main(int argc, char *argv[]) {
    int shape0 = 1; // TO MODIFY before run
    int shape1 = 4096; // TO MODIFY before run
    
    int in_size = shape0 * shape1;

    float* d_in = (float*) malloc(sizeof(float) * in_size);
    float* d_in_trans = (float*) malloc(sizeof(float) * in_size);

    loadWeights(d_in, "/home/data/onellm/0_self_decoder_qk_v_after_bmm.bin", shape0, shape1); // TO MODIFY
    loadWeights_trans(d_in_trans, "/home/data/trans/self_decoder_qk_v_buf_after_bmm_trans.bin", shape0, shape1); // TO MODIFY
    std::cout << "====intermediate tensor comparison result====" << "\n";
    CheckResult(d_in, d_in_trans, shape0 * shape1);

    free(d_in);
    free(d_in_trans);

}
