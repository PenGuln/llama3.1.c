## Getting started

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone https://github.com/PenGuln/llama3.1.c.git
```

Then, open the repository folder:
```bash
cd llama3.1.c
```

Now, let's chat with llama3 in pure C. Firstly download the Llama3.1-Instruct checkpoint from huggingface:
```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="meta-llama/Llama-3.1-8B-Instruct", allow_patterns="original/*", local_dir="./")
```
rename the checkpoint folder to `Llama-3.1-8B-Instruct`.

```bash
mv original Llama-3.1-8B-Instruct
```

Then export the weight and quantized weight(optional)
```bash
python export.py llama3.1_8b_instruct.bin --meta-llama Llama-3.1-8B-Instruct
python export.py llama3.1_8b_instruct_quant.bin --meta-llama Llama-3.1-8B-Instruct --version 2
```

Export the tokenizer
```bash
python tokenizer.py --tokenizer-model=Llama-3.1-8B-Instruct/tokenizer.model
mv Llama-3.1-8B-Instruct/tokenizer.bin  ./tokenizer.bin
```

Starting chat
```bash
make run
./run llama3.1_8b_instruct.bin -z tokenizer.bin -m chat
```

or in quantization mode
```bash
make run
./runq llama3.1_8b_instruct_quant.bin -z tokenizer.bin -m chat
```

## Evaluation of Inference speed

Prepare test prompts from ultrachat-200K.
```bash
python ultrachat.py
```

Then a `data.bin` will appear in the repository folder. Evaluate the average inference speed on the dataset by running
```bash
./test llama3.1_8b_instruct.bin -z tokenizer.bin -n 128
```

To test the openMP complied inference speed, you'll need to complile with
```bash
make runomp
```

Then run the same script for evaluation.
```bash
./test llama3.1_8b_instruct.bin -z tokenizer.bin -n 128
```