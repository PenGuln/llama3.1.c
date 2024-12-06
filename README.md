## getting started

First, navigate to the folder where you keep your projects and clone this repository to this folder:

```bash
git clone
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

```bash
mv original Llama-3.1-8B-Instruct
```

Then export the weight
```bash
python export.py llama3.1_8b_instruct.bin --meta-llama Llama-3.1-8B-Instruct
```

Export the tokenizer.bin
```bash
python tokenizer.py --tokenizer-model=Llama-3.1-8B-Instruct/tokenizer.model
```

Starting chat
```bash
make run
./run llama3_8b_instruct.bin -z tokenizer_llama3.bin -m chat
```