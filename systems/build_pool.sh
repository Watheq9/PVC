#!/bin/bash

# Define the list of transcription models
models=(\
        "spotify" \
        "silero-small" \
        "whisperX-large-v3" \
        "silero-large" \
        "whisperX-base" \
        ) #  \


export PYTHONPATH=$PYTHONPATH:$(pwd)
project_dir="path_to_your_project"
data_dir="${project_dir}/data"
runs_dir="${project_dir}/data/runs/pool"
converted_runs_dir="${project_dir}/data/runs/pool_rankZephyr/monoT5_converted"

query_file="${project_dir}/data/queries/podcasts_query_variations_sample.tsv"
query_file="${project_dir}/data/queries/podcasts_query_variations.tsv"
experiment="query_variations"


# set the flag to 1 to run its corresponding search block
SPARSE=0
RM3=0
COLBERT=0
SPLADE=0
DENSE=0
MONOT5=0
RANK_ZEPHUR=1


# ----- lexical retrieval configurations --------------------
if [ $SPARSE -eq 1 ]; then

seg_length=120
seg_step=60
index_type='one-field'
retrieval_depth=200
retr_models=(\
        "BM25" \
        "PL2" \
        "DPH" \
        )

for retr_model in "${retr_models[@]}"; do
    for tr_model in "${models[@]}"; do
        echo "Run retrieval using the index from ${tr_model}"

        index_path="${project_dir}/data/indexes/terrier/${tr_model}_${seg_length}_${seg_step}_${index_type}_time_segment/data.properties"
        log_file="${data_dir}/logs/pool/${tr_model}_${retr_model}.log"
        run_path="${runs_dir}/${tr_model}_${retr_model}.tsv"

        python systems/sparse/lexical_retrieval.py \
            --index_path ${index_path} \
            --run_path ${run_path} \
            --query_file ${query_file} \
            --model ${retr_model} \
            --depth ${retrieval_depth} 2>&1 | tee -a ${log_file}
    done
done
fi

# ----- BM25 + RM3 retrieval configurations --------------------
if [ $RM3 -eq 1 ]; then

seg_length=120
seg_step=60
index_type='one-field'
retrieval_depth=200
retr_models=(\
        "BM25+RM3")

for retr_model in "${retr_models[@]}"; do
    for tr_model in "${models[@]}"; do
        echo "Run retrieval using the index from ${tr_model}"

        index_path="${project_dir}/data/indexes/terrier/${tr_model}_${seg_length}_${seg_step}_${index_type}_time_segment/data.properties"
        log_file="${data_dir}/logs/pool/${tr_model}_${retr_model}.log"
        run_path="${runs_dir}/${tr_model}_${retr_model}.tsv"

        python systems/sparse/RM3.py \
            --index_path ${index_path} \
            --run_path ${run_path} \
            --query_file ${query_file} \
            --model ${retr_model} \
            --depth ${retrieval_depth} 2>&1 | tee -a ${log_file}
    done
done
fi


# ----- ColBERT configurations --------------------
if [ $COLBERT -eq 1 ]; then
GPU_NUMBER=0
nbits=2   # encode each dimension with 2 bits
doc_maxlen=500   # truncate passages at 500 tokens
index_bsize=256
colbert_model="colbertv2.0"
topk=100

# conda activate colbert

for tr_model in "${models[@]}"; do

    echo "Running ColBERT retrieval for ${tr_model} using query file from ${query_file}"

    index_dir="${data_dir}/indexes/${tr_model}"
    corpus="${data_dir}/tsv_corpus/${tr_model}_120_60_time_segment.tsv"
    run_path="${runs_dir}/${tr_model}_${colbert_model}_topk${topk}.tsv"
    log_file="${data_dir}/logs/pool/${tr_model}_${colbert_model}_${experiment}.log"
    log_t="${data_dir}/logs/pool/${tr_model}_${colbert_model}_${experiment}.txt"

    CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python systems/colBERT/retrieve.py \
        --query_path ${query_file}\
        --corpus_path ${corpus}\
        --colbert_name ${colbert_model} \
        --run_path ${run_path} \
        --index_dir ${index_dir}\
        --topk ${topk}\
        --log_file ${log_file} 2>&1 | tee -a ${log_t}

    echo "Done retrieval"
done
fi




# ----- SPLADE configurations --------------------
if [ $SPLADE -eq 1 ]; then
splade_dir="path_to_splade_repo"
cd ${splade_dir}
source /home/watheq/anaconda3/etc/profile.d/conda.sh
conda activate splade

export PYTHONPATH=$PYTHONPATH:$(pwd)
export SPLADE_CONFIG_NAME="spotify_spladepp_cocondenser_ensembledistil_monogpu.yaml"
splade_models=("splade-cocondenser-ensembledistil" "splade-v3")
top_k=100

for tr_model in "${models[@]}"; do
    for splade_model in "${splade_models[@]}"; do
        echo "Running retrieval for ${tr_model} with Splade ${splade_model}"
        model_dir="/home/watheq/.cache/huggingface/hub/${splade_model}" #or you can use "naver/${splade_model}"
        index_dir="${data_dir}/indexes/${tr_model}/${splade_model}"
        corpus="${data_dir}/tsv_corpus/${tr_model}_120_60_time_segment.tsv"
        log_t="${data_dir}/logs/pool/${tr_model}_${splade_model}.txt"

        CUDA_VISIBLE_DEVICES=1 python3 -m splade.retrieve \
                    init_dict.model_type_or_dir=${model_dir} \
                    config.pretrained_no_yamlconfig=true \
                    config.index_dir=${index_dir} \
                    config.out_dir=${runs_dir} \
                    config.top_k=${top_k} \
                    config.splade_model=${splade_model} \
                    config.retrieval_name="${tr_model}_${splade_model}" \
                    data.Q_COLLECTION_PATH=${query_file} \
                    data.COLLECTION_PATH=$corpus 2>&1 | tee -a ${log_t}

        echo "Done retrieval"
    done
done
fi


# ----------- Dense configurations ------------------------

if [ $DENSE -eq 1 ]; then

source /home/watheq/anaconda3/etc/profile.d/conda.sh
conda activate flag # environment to use bge models
embed_models=("BAAI/bge-large-en-v1.5" "BAAI/bge-m3")
dense_names=("bge-large-en-v1.5" "bge-m3")
topk=100
search_type="flat"

for transcription_model in "${models[@]}"; do
    for i in "${!embed_models[@]}"; do

        embed_model=${embed_models[i]}
        model_name=${dense_names[i]}

        echo "Running dense retrieval for tr_model ${transcription_model} using ${model_name}"
        embed_file="${project_dir}/data/dense/embeddigs/${transcription_model}_embed_using_${model_name}.pkl"
        run_file="${runs_dir}/${transcription_model}_${model_name}.tsv"
        log_file="${data_dir}/logs/pool/${transcription_model}_${model_name}.txt"

        CUDA_VISIBLE_DEVICES=1 python systems/dense_search.py \
            --query_file ${query_file} \
            --model_name  ${embed_model} \
            --embed_file ${embed_file} \
            --topk ${topk} \
            --output ${run_file} \
            --log ${log_file} \
            --search_type ${search_type} \
            --apply_post_processing 2>&1 | tee -a ${log_file}


        echo "Done"
        echo "====================================================="
    done
done
fi



# ----------- monoT5 configurations ------------------------
if [ $MONOT5 -eq 1 ]; then
monot5_dir="path_to_llm-rankers_repo"
cd ${monot5_dir}
source /home/watheq/anaconda3/etc/profile.d/conda.sh
conda activate llm-ranker

systems=(\
        "colbertv2.0_" \
        "splade-v3." \
        "BM25." \
        "BM25+RM3." \
        "DPH." \
        ) #  \
topk=100
seg_length=120
step=60
query_length=25
passage_length=475
batch_size=32
monoT5="monoT5"

for transcription_model in "${models[@]}"; do
    for system in "${systems[@]}"; do
        for initial_run in $(find ${runs_dir} -type f -name "${transcription_model}_${system}*"); do

            reranked_run="${runs_dir}/${transcription_model}_${system}+${monoT5}.tsv"
            input_segments="${project_dir}/data/dataset_files/${transcription_model}_${seg_length}_${step}_time_segment.jsonl"
            log_file="${data_dir}/logs/pool/${transcription_model}_${system}+${monoT5}.txt"

            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Reranking a run for ${transcription_model} from ${initial_run} and topk = ${topk}"  | tee -a ${log_file}
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] reranked run will be saved to ${reranked_run}"  | tee -a ${log_file}
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] log_file  ${log_file}"  | tee -a ${log_file}
            CUDA_VISIBLE_DEVICES=0 python run.py run --model_name_or_path castorini/monot5-base-msmarco --tokenizer_name_or_path castorini/monot5-base-msmarco  --run_path ${initial_run} --save_path ${reranked_run} --dataset_file ${input_segments} --queries_file ${query_file} --hits ${topk} --query_length ${query_length} --passage_length ${passage_length} --device cuda pointwise --method yes_no --batch_size ${batch_size} 2>&1 | tee -a ${log_file}


            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done reranking and run was saved to ${reranked_run}"  | tee -a ${log_file}
        done
    done
done
fi


# ----------- RankZephur configurations ------------------------


if [ $RANK_ZEPHUR -eq 1 ]; then

source /home/watheq/anaconda3/etc/profile.d/conda.sh
conda activate rankllm

systems=(\
        "colbertv2.0+monoT5" \
        "splade-v3+monoT5" \
        "BM25+monoT5" \
        # "BM25+RM3+monoT5" \ 
        # "DPH+monoT5" \
        ) #  \


topk=40
seg_length=120
step=60 

llm_model=castorini/rank_zephyr_7b_v1_full
window_size=20
context_size=4096
RankZephyr="RankZephyr"

for transcription_model in "${models[@]}"; do
    for system in "${systems[@]}"; do
        for initial_run in $(find ${runs_dir} -type f -name "${transcription_model}_${system}*"); do

            # 1. convert the initial retrieval run from trec format to format suitable for rank-llm library
            corpus="${project_dir}/data/dataset_files/${transcription_model}_${seg_length}_${step}_time_segment.jsonl"

            converted_run="${converted_runs_dir}/${transcription_model}_${system}+${RankZephyr}.jsonl"
            reranked_run="${transcription_model}_${system}+${RankZephyr}"
            log_file="${data_dir}/logs/pool/${transcription_model}_${system}+${RankZephyr}.txt"

            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Converting a run for ${transcription_model} system= ${system} from ${initial_run} topk=${topk}" | tee -a ${log_file}
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Converted run will be saved to ${converted_run}" 
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] log_file  ${log_file}"

            python systems/RankZephyr/convert_run_to_rankllm_format.py  --input ${initial_run} --topk ${topk} --query ${query_file} \
                                                            --corpus ${corpus} --output ${converted_run}


        # 2. Run the Rank-Zephur (llm-based list-wise) reranker using the converted run
        for converted_run in $(find ${converted_runs_dir} -type f -name "${transcription_model}_${system}*"); do

            reranked_run="${transcription_model}_${system}+${RankZephyr}"
            log_file="${data_dir}/logs/pool/${transcription_model}_${system}+${RankZephyr}.txt"

            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Reranking a run for ${transcription_model} system= ${system} from ${converted_run} topk=${topk}" | tee -a ${log_file}
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] reranked run will be saved to ${runs_dir}/${reranked_run}" | tee -a ${log_file}
            echo "[$(date +'%Y-%m-%d %H:%M:%S')] log_file  ${log_file}"

            CUDA_VISIBLE_DEVICES=1 python systems/RankZephyr/run_reranker.py \
                 --model_path=${llm_model}  --top_k_candidates=${topk} --input_run=${converted_run}  \
                 --retrieval_method=${reranked_run} --prompt_mode=rank_GPT  --context_size=${context_size} \
                 --variable_passages --window_size ${window_size} --save_dir ${runs_dir} 2>&1 | tee -a ${log_file}

            echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done reranking" | tee -a ${log_file}
        done
    done
done 

fi
