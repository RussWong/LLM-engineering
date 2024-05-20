from transformers import AutoTokenizer, LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("/home/gyou_llama/llama/out_7B")
tokenizer = AutoTokenizer.from_pretrained("/home/gyou_llama/llama/out_7B")
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
generate_ids = model.generate(inputs.input_ids, max_length=30)
