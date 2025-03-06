import time
import argparse
import os
import sys
import copy
import torch
from enum import Enum
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
# parent = os.path.dirname(parent)
sys.path.append(parent)

# print(parent)

from RankZephyr.rankllm import PromptMode
from RankZephyr.reranker import IdentityReranker, RankLLM, Reranker
from RankZephyr.data import Query, Request, read_requests_from_file
from typing import Any, Dict, List, Union



class RetrievalMode(Enum):
    DATASET = "dataset"
    CUSTOM = "custom"

    def __str__(self):
        return self.value



def rerank(
    model_path: str,
    input_run: str,
    retrieval_method: str,
    top_k_retrieve: int = 50,
    top_k_rerank: int = 10,
    shuffle_candidates: bool = False,
    print_prompts_responses: bool = False,
    num_passes: int = 1,
    interactive: bool = False,
    default_agent: RankLLM = None,
    rerank_dir="",
    **kwargs: Any,
):

    """
    Rerank a given run 
    rerank_dir: directory to save the reranked run in jsonl and txt format

    Returns:
        - List of top_k_rerank candidates
    """

    # Get reranking agent
    reranker = Reranker(
        Reranker.create_agent(model_path.lower(), default_agent, interactive, **kwargs)
    )

    # Retrieve initial candidates
    print(f"Reading the input file run from {input_run} ...")
    requests = read_requests_from_file(input_run)

    # Reranking stages
    print(f"Reranking and returning {top_k_rerank} documents with {model_path}...")
    if reranker.get_agent() is None:
        # No reranker. IdentityReranker leaves retrieve candidate results as is or randomizes the order.
        shuffle_candidates = True if model_path == "rank_random" else False
        rerank_results = IdentityReranker().rerank_batch(
            requests,
            rank_end=top_k_retrieve,
            shuffle_candidates=shuffle_candidates,
        )
    else:
        # Reranker is of type RankLLM
        for pass_ct in range(num_passes):
            print(f"Pass {pass_ct + 1} of {num_passes}:")

            rerank_results = reranker.rerank_batch(
                requests,
                rank_end=top_k_retrieve,
                rank_start=0,
                shuffle_candidates=shuffle_candidates,
                logging=print_prompts_responses,
                top_k_retrieve=top_k_retrieve,
                **kwargs,
            )

        if num_passes > 1:
            requests = [Request(copy.deepcopy(r.query), copy.deepcopy(r.candidates)) for r in rerank_results]

    for rr in rerank_results:
        rr.candidates = rr.candidates[:top_k_rerank]

    file_name = reranker.write_rerank_results(
            retrieval_method,
            rerank_results,
            shuffle_candidates,
            top_k_candidates=top_k_retrieve,
            pass_ct=None if num_passes == 1 else pass_ct,
            rerank_results_dirname=rerank_dir,
            window_size=kwargs.get("window_size", None),
        )

    print(file_name)


    if interactive:
        return (rerank_results, reranker.get_agent())
    else:
        return rerank_results




def main(args):
    model_path = args.model_path
    batch_size = args.batch_size
    use_azure_openai = args.use_azure_openai
    context_size = args.context_size
    top_k_candidates = args.top_k_candidates
    top_k_rerank = top_k_candidates if args.top_k_rerank == -1 else args.top_k_rerank
    input_run = args.input_run
    num_gpus = args.num_gpus
    retrieval_method = args.retrieval_method
    prompt_mode = args.prompt_mode
    num_few_shot_examples = args.num_few_shot_examples
    shuffle_candidates = args.shuffle_candidates
    print_prompts_responses = args.print_prompts_responses
    num_few_shot_examples = args.num_few_shot_examples
    device = "cuda" if torch.cuda.is_available() else "cpu"
    variable_passages = args.variable_passages
    retrieval_mode = RetrievalMode.CUSTOM
    num_passes = args.num_passes
    step_size = args.step_size
    window_size = args.window_size
    system_message = args.system_message
    vllm_batched = args.vllm_batched
    save_dir = args.save_dir

    _ = rerank(
        model_path=model_path,
        batch_size=batch_size,
        input_run=input_run,
        retrieval_mode=retrieval_mode,
        retrieval_method=retrieval_method,
        top_k_retrieve=top_k_candidates,
        top_k_rerank=top_k_rerank,
        context_size=context_size,
        device=device,
        num_gpus=num_gpus,
        prompt_mode=prompt_mode,
        num_few_shot_examples=num_few_shot_examples,
        shuffle_candidates=shuffle_candidates,
        print_prompts_responses=print_prompts_responses,
        use_azure_openai=use_azure_openai,
        variable_passages=variable_passages,
        num_passes=num_passes,
        window_size=window_size,
        step_size=step_size,
        system_message=system_message,
        vllm_batched=vllm_batched,
        rerank_dir=save_dir,
    )


""" sample run:
python src/rank_llm/scripts/run_reranker.py  --model_path=castorini/rank_zephyr_7b_v1_full  --top_k_candidates=100 --input_run="/storage/users/watheq/projects/podcast_search/data/runs/time_segments/2020/rank_llm/silero-large_rankllm_run_top100_segLen_120.jsonl"  --retrieval_method=BM25 --prompt_mode=rank_GPT  --context_size=4096 --variable_passages --window_size 20



"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model. If `use_azure_ai`, pass your deployment name.",
    )
    parser.add_argument(
        "--input_run",
        type=str,
        required=True,
        help="Path to the input run in rank-llm format",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Size of each batch for batched inference.",
    )
    parser.add_argument(
        "--use_azure_openai",
        action="store_true",
        help="If True, use Azure OpenAI. Requires env var to be set: "
        "`AZURE_OPENAI_API_VERSION`, `AZURE_OPENAI_API_BASE`",
    )
    parser.add_argument(
        "--context_size", type=int, default=4096, help="context size used for model"
    )
    parser.add_argument(
        "--top_k_candidates",
        type=int,
        default=100,
        help="the number of top candidates to rerank",
    )
    parser.add_argument(
        "--top_k_rerank",
        type=int,
        default=-1,
        help="the number of top candidates to return from reranking",
    )
    parser.add_argument(
        "--num_gpus", type=int, default=1, help="the number of GPUs to use"
    )
    parser.add_argument(
        "--retrieval_method",
        type=str,
        required=True,
        help=f"Name of the method used to add it to the resutled run",
    )
    parser.add_argument(
        "--prompt_mode",
        type=PromptMode,
        required=True,
        choices=list(PromptMode),
    )
    parser.add_argument(
        "--shuffle_candidates",
        action="store_true",
        help="whether to shuffle the candidates before reranking",
    )
    parser.add_argument(
        "--print_prompts_responses",
        action="store_true",
        help="whether to print promps and responses",
    )
    parser.add_argument(
        "--num_few_shot_examples",
        type=int,
        required=False,
        default=0,
        help="number of in context examples to provide",
    )
    parser.add_argument(
        "--variable_passages",
        action="store_true",
        help="whether the model can account for variable number of passages in input",
    )
    parser.add_argument(
        "--num_passes",
        type=int,
        required=False,
        default=1,
        help="number of passes to run the model",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="window size for the sliding window approach",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step size for the sliding window approach",
    )
    parser.add_argument(
        "--system_message",
        type=str,
        default="You are RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query.",
        help="the system message used in prompts",
    )
    parser.add_argument(
        "--vllm_batched",
        action="store_true",
        help="whether to run the model in batches",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="Zephur_reranked_run",
        help="The directory to save the reranked run in ",
    )
    args = parser.parse_args()
    main(args)
