from datasets import load_dataset

# TEMPLATE = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
data = load_dataset("HuggingFaceH4/ultrachat_200k", "default", split="train_sft")
data = data.shuffle(seed = 42)
data = data["messages"][:1]
  
f = open("data.bin", "wb")
for item in data:
    res = bytes(item[0]['content'], 'utf-8')
    f.write(res)
    f.write(b'\0')
f.close()

