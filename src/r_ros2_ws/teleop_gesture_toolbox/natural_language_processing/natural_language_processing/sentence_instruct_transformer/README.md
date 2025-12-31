# Sentence Instruct Transformer



## Install 

```
conda create --name clm
conda activate clm
conda install -c conda-forge transformers pytorch accelerate -c pytorch
cd <this repo>
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct qwen_model
```

## Run test

Currently passes following test: `python test_nlp.py`
