#include <stdio.h>
#include "src/utils/model_utils.h"

struct ConvertedModel {
    std::string model_path = "/home/llamaweight/"; // 模型文件路径
    std::string tokenizer_path = "/home/llama2-7b-tokenizer.bin"; // tokenizer文件路径
};

int main(int argc, char **argv) {
    int round = 0;
    std::string history = "";
    ConvertedModel model;
    // auto model = llm::CreateDummyLLMModel<float>(model.tokenizer_file);//load dummy weight + load tokenizer
    auto llm_model = llm::CreateRealLLMModel<float>(model.model_path, model.tokenizer_path);//load real weight + load tokenizer
    std::string model_name = llm_model->model_name;
    // exist when generate end token or reach max seq
    while (true) {
        printf("please input the question: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "s") {//停止对话
            break;
        }    
        // (RussWong)notes: index = 生成的第几个token，从0开始
        std::string retString = llm_model->Response(llm_model->MakeInput(history, round, input), [model_name](int index, const char* content) {
            if (index == 0) {
                printf(":%s", content);
                fflush(stdout);
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        });
        //(RussWong)notes: 多轮对话保留history，和当前轮次input制作成新的上下文context
        history = llm_model->MakeHistory(history, round, input, retString);
        round++;
    }
    return 0;
}
