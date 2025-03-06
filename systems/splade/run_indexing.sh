#!/bin/bash

# Define the list of models
models=(\
        "spotify" \
        "whisperX-base" \
        "whisperX-large-v3" \
        "silero-large" \
        "silero-small" \
        ) #  \

export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="spotify_spladepp_cocondenser_ensembledistil_monogpu.yaml"
# experiment 1
# splade_model="splade-cocondenser-ensembledistil"
# model_dir="naver/${splade_model}"

# experiment 2
splade_model="splade-v3"
model_dir="/home/watheq/.cache/huggingface/hub/${splade_model}"
data_dir="/storage/users/watheq/projects/podcast_resource/data"

for tr_model in "${models[@]}"; do

    echo "Building the index for ${tr_model}"
    index_dir="${data_dir}/indexes/${tr_model}/${splade_model}"
    run_dir="${data_dir}/runs/${tr_model}/${splade_model}"
    corpus="${data_dir}/tsv_corpus/${tr_model}_120_60_time_segment.tsv"
    log_t="${data_dir}/logs/${tr_model}_${splade_model}.txt"

    CUDA_VISIBLE_DEVICES=1 python3 -m splade.index \
    init_dict.model_type_or_dir=${model_dir} \
    config.pretrained_no_yamlconfig=true \
    config.index_dir=${index_dir} \
    data.COLLECTION_PATH=$corpus 2>&1 | tee -a ${log_t}

    echo "Done indexing" 
done

# conda activate splade
# chmod +x run_indexing.sh
# ./run_indexing.sh