

1. Create a conda environment

```
conda create -n llamacpp python=3.10
conda activate llamacpp
conda install nvidia/label/cuda-12.5.0::cuda
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentencepiece safetensors protobuf huggingface-hub clean-text[gpl] pandas numpy tqdm toml==0.10.2 openai-1.65.4 tiktoken-0.9.0 


export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# install llamacpp python library
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

```


2.  Form the pairs out of the created pool to be judged by LLM.


Edit the data paths within ``form_pool_pairs.ipynb`` and then run it.



3. Run the ``judge_pairs.sh`` script to judge the pairs generated in the previous step. You need to edit the data paths accordingly.