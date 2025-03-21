from transformers import AutoTokenizer, LlamaForCausalLM
# 注意，这个py用来debug的，作为本课程各个kernel的groudtruth，且此huggingface接口只接受Llama-2-7b-hf，不接受Llama-2-7b
# 注意，此脚本我只在pip install transformers==4.38 sentencepiece accelerate的环境下验证了能跑通并拿来作为groudtruth
model = LlamaForCausalLM.from_pretrained("/path/to/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("/path/to/Llama-2-7b-hf")
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
