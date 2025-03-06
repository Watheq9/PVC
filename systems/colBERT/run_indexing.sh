#!/bin/bash

# Define the list of models
models=(\
        "spotify" \
        "whisperX-base" \
        "whisperX-large-v3" \
        "silero-large" \
        "silero-small" \
        ) #  \

GPU_NUMBER=0
nbits=2   # encode each dimension with 2 bits
doc_maxlen=500   # truncate passages at 500 tokens
index_bsize=256
export PYTHONPATH=$PYTHONPATH:$(pwd)

colbert_model="colbertv2.0"
project_dir="path_to_your_project"
checkpoint="${project_dir}/models/${colbert_model}"
data_dir="${project_dir}/data"


for tr_model in "${models[@]}"; do

    echo "Building the index for ${tr_model}"

    index_dir="${data_dir}/indexes/${tr_model}/${colbert_model}"
    run_dir="${data_dir}/runs/${tr_model}/${colbert_model}"
    corpus="${data_dir}/tsv_corpus/${tr_model}_120_60_time_segment.tsv"
    log_file="${data_dir}/logs/colBERT/${tr_model}_${colbert_model}.log"
    log_t="${data_dir}/logs/colBERT/${tr_model}_${colbert_model}.txt"


    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python systems/colBERT/index.py \
        --corpus_path ${corpus}\
        --checkpoint ${checkpoint}\
        --nbits ${nbits} \
        --doc_maxlen ${doc_maxlen}\
        --index_dir ${index_dir}\
        --index_bsize ${index_bsize}\
        --log_file ${log_file} 2>&1 | tee -a ${log_t}

    echo "Done indexing" 
done

# conda activate colbert
# chmod +x run_indexing.sh
# ./run_indexing.sh

