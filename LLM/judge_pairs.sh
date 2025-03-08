#!/bin/bash


ANNAPURNA=0 # default server
MIGHTY_MOUSE=1
BUNYA=2 
SERVER=$ANNAPURNA

# ------ annapurna  ----------
project_dir="your_project_dir"
models_dir="${project_dir}/models/LLM"
pairs_dir="${project_dir}/data/runs/pool_rrf/st5_judgement_pairs/one_file"


# ---------------- LLM list -------------------------
MISTRAL_Q6="${models_dir}/Mistral/Mistral-Small-Instruct-2409-Q6_K_L.gguf"
CALME="${models_dir}/calme-3.2-instruct-78b.Q4_K_M.gguf"
QWEN_Q8="${models_dir}/Qwen2.5-14B-Instruct-Q8_0.gguf"
LLAMA3="${models_dir}/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
GEMMA2="${models_dir}/gemma-2-9b-it-Q8_0.gguf"


# Define the list of transcription models
tr_models=(\
        "spotify" \
        "silero-small" \
        "whisperX-large-v3" \
        "silero-large" \
        "whisperX-base" \
        ) #  \

declare -A llm_models
llm_models[$MISTRAL_Q6]="MISTRAL_Q6" 
llm_models[$QWEN_Q8]="QWEN_Q8" 
llm_models[$LLAMA3]="LLAMA3" 
llm_models[$GEMMA2]="GEMMA2" 


log_dir="${project_dir}/data/logs"
prompt_template="${project_dir}/LLM/prompts/pool_prompt_template.toml"
cd ${project_dir}



# 1. Judge pool with open-source LLMs
for tr_model in "${tr_models[@]}"; do
for LLM_MODEL in "${!llm_models[@]}"; do
for pairs_file in ${pairs_dir}/${tr_model}*; do
        log_file="${log_dir}/${tr_model}_all.log"   
        filename="${pairs_file##*/}"    # Get the basename (example.txt)
        result_name="${filename%.*}"     # Remove the extension
        LLM_NAME=${llm_models[$LLM_MODEL]}
        result_file="${project_dir}/data/LLM/${tr_model}_judged_with_${LLM_NAME}.jsonl"
    
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Processing: $pairs_file" | tee -a ${log_file}
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Saving the result to ${result_file}" | tee -a ${log_file}
        CUDA_VISIBLE_DEVICES=1 python ${project_dir}/LLM/llm_judge.py \
            --model ${LLM_MODEL} \
            --prompt_template ${prompt_template} \
            --input    ${pairs_file}\
            --result   ${result_file} \
            --log_file ${log_file}  2>&1 | tee -a  ${log_file}

        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Done Judging" | tee -a ${log_file}

done
done
done




# 2. Judge pool with GPT4-o

for tr_model in "${tr_models[@]}"; do


    model="gpt-4o-2024-11-20"
    output_mode="object"
    prompt_type="user_system"
    prompt_name="pool_prompt_template"
    experiment="${prompt_name}-GPT4o_${tr_model}"
    prompt_template="${project_dir}/LLM/prompts/${prompt_name}.toml"
    log="${project_dir}/data/logs/${tr_model}/${experiment}.txt"
    input="${project_dir}/data/pool/judgement_pairs/${tr_model}_pairs.jsonl"
    result="${project_dir}/data/pool/llm_judged/${tr_model}_judged_with_GPT4o.jsonl"

    python LLM/gpt_judge.py\
        --model ${model} \
        --input ${input} \
        --prompt_template ${prompt_template} \
        --result ${result} \
        --log_file ${log} \
        --output_mode ${output_mode} \
        --prompt_type ${prompt_type} 

done




