import os, sys
sys.path.append('../') 
import pandas as pd
import argparse
import json


def convert_run_to_rankllm_format(input_run, save_file, corpus, query, topk,
                                  text_col='seg_words'):
    
    written = 0  # number of written lines
    df_queries = pd.read_csv(query, sep='\t', dtype={"qid": "str", "query": "str"}, names=['qid', 'query'],)
    df_corpus = pd.read_json(corpus, lines=True)

    doc_store = df_corpus.set_index('id').to_dict(orient='index')
    query_store = df_queries.set_index('qid').to_dict(orient='index')

    df_run = pd.read_csv(input_run, sep=r'\s+', names=["qid", "Q0", "docid", "rank", 'score', 'tag'], 
                       dtype={"qid": "str", "docid": "str"})

    
    print(f"Number of queries retrieved in the run {len(df_run['qid'].unique())}")
    
    # Start new empty file
    with open(save_file, 'w') as file:
        file.write('') 

    grouped = df_run.groupby('qid')
    for qid, df_q in grouped:

    # for qid in df_run['qid'].unique():
    #     df_q =  df_run[df_run['qid'] == qid]

        line ={
            "query":{"text": query_store.get(qid).get("query"), "qid": qid},
            "candidates":[]}
        
        # for i, row in enumerate(qid_rows.itertuples()):
        cnt = 0
        missed = []
        for index, row in df_q.iterrows():
            if cnt >= topk: # only append topk documents
                break
            # # row = row._asdict
            # print(type(row))
            # print(row.columns)
            docid = row['docid']
            score = row['score']
            if doc_store.get(docid) is None:
                print("None docid ", docid)
                missed.append(docid)
            else:
                doc_text = doc_store.get(docid).get(text_col)
                candidate = {"docid": docid, "score": score, "doc":{"contents": doc_text}}
                line['candidates'].append(candidate)
            cnt += 1
            # dictionary format:
            # { "query":{ "text":"are naturalization records public information", "qid": 23849},
            #   "candidates":[
            #       {"docid":"8010561", "score":114183.0, "doc":{"contents": "document text"},
            #       {"docid":"8010511", "score":114858.0, "doc":{"contents": "document text 2"}]
 
        try: # write the line
            with open(save_file, 'a') as file:
                file.write(json.dumps(line) + '\n')  # Write the JSON line to the file
            written += 1
        except Exception as e:
            raise Exception("Writing error: {}".format(e))

    print(f"Number of written queries {written} out of {len(df_queries)}")
    # assert written == len(df_queries)



def get_input_parameters():
    transcription_model="silero-small" 
    topk=100
    length=120
    year=2020
    step=60 
    query=f"/storage/collections/trec-podcast/corpus-trec-podcasts-2020/podcasts-no-audio-13GB/TREC/topics/podcasts_2020_queries_test.tsv"

    first_stage_runs_dir=f"/storage/users/watheq/projects/podcast_search/data/runs/time_segments/{year}/first_stage"
    converted_runs_dir=f"/storage/users/watheq/projects/podcast_search/data/runs/time_segments/{year}/llm_converted_runs"
    reranked_run_save_dir=f"/storage/users/watheq/projects/podcast_search/data/runs/time_segments/2020/zephyr_reranked"

    llm_model="castorini/rank_zephyr_7b_v1_full"
    window_size=20
    context_size=4096
    experiment=f"full_corpus-1"
    # conda activate rankllm

    # 1. convert the initial retrieval run from trec format to format suitable for rank-llm library
    input_run=f"{first_stage_runs_dir}/{transcription_model}_{length}_{step}.run"
    converted_run=f"{converted_runs_dir}/{transcription_model}_rankllm_run_top{topk}_segLen_{length}_{experiment}.jsonl"
    corpus=f"/storage/users/watheq/projects/podcast_search/data/dataset_files/{transcription_model}_{length}_{step}_time_segment.jsonl"

    # python reranking/convert_run_to_rankllm_format.py  --input {input_run} --topk {topk} --query {query} \
                                                # --corpus {corpus} --output {converted_run}

    return input_run, query, topk, corpus, converted_run




def main():
    parser = argparse.ArgumentParser(description="Script to convert a TREC-format run to jsonl file to be reranked by RankLLM model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input run file")
    parser.add_argument("--topk", type=int, required=True, help="Topk documents to be reranked to the input run file")
    parser.add_argument("--query", type=str, required=True, help="Path to the query file")
    parser.add_argument("--corpus", type=str, required=True, help="Path to the corpus to read the document from")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output file with .jsonl extension")
    args = parser.parse_args()
    input = args.input
    query = args.query
    topk = args.topk
    corpus = args.corpus
    output = args.output

    # input, query, topk, corpus, output = get_input_parameters()

    convert_run_to_rankllm_format(input_run=input, save_file=output, corpus=corpus, query=query, 
                                  topk=topk, text_col='seg_words')


if __name__ == "__main__":
    main()


