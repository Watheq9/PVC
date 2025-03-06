#!/bin/bash

# Define the list of models
models=(\
        "spotify" \
        "whisperX-base" \
        "silero-large" \
        "silero-small" \
        "whisperX-large-v3" \
        ) #  \
years=(2020 2021)

GPU_NUMBER=0
nbits=2   # encode each dimension with 2 bits
topk=1000   
export PYTHONPATH=$PYTHONPATH:$(pwd)

colbert_model="colbertv2.0"
project_dir="path_to_your_project"
data_dir="${project_dir}/data"


for tr_model in "${models[@]}"; do
    for year in "${years[@]}"; do

        echo "Running retrieval for ${tr_model}"

        index_dir="${data_dir}/indexes/${tr_model}"
        corpus="${data_dir}/tsv_corpus/${tr_model}_120_60_time_segment.tsv"
        log_file="${data_dir}/logs/colBERT/retrieval_of_${tr_model}_${colbert_model}.log"
        log_t="${data_dir}/logs/colBERT/retrieval_of_${tr_model}_${colbert_model}.txt"
        run_path="${data_dir}/runs/${year}/${tr_model}/${tr_model}.${colbert_model}-topk${topk}-query${year}.tsv"
        query_path="${data_dir}/queries/podcasts_${year}_topics_test.tsv"


        CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python systems/colBERT/retrieve.py \
            --query_path ${query_path}\
            --corpus_path ${corpus}\
            --colbert_name ${colbert_model} \
            --run_path ${run_path} \
            --index_dir ${index_dir}\
            --topk ${topk}\
            --log_file ${log_file} 2>&1 | tee -a ${log_t}

        echo "Done retrieval" 
    done
done

# conda activate colbert
# chmod +x systems/colBERT/run_retrieval.sh
# ./systems/colBERT/run_retrieval.sh


    


