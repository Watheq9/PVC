import os, json
import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import pandas as pd
import argparse
import logging
import os
import pickle
from collections import OrderedDict


os.environ["NCCL_DEBUG"] = "INFO" # giving you a more helpful error message to google.



def initailize_logger(logger, log_file, level):

    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    if not len(logger.handlers):  # avoid creating more than one handler
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(formatter)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)

    return logger


def get_logger(log_file="progress.txt", level=logging.DEBUG):
    """Returns a logger to log the info and error messages in an organized and timestampped way"""

    logger = logging.getLogger(log_file)
    logger = initailize_logger(logger, log_file, level)
    return logger


def save_dict(file_name, my_dict):
    '''
    save_file: path to save the dictionary with .json extension
    '''
    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        json.dump(my_dict, f)


def save_list(file_name, data_list, method='json'):
    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

    if method == 'json':
        with open(file_name, 'w') as json_file:
            json.dump(data_list, json_file)

    else: # pickling
        with open(file_name, "wb") as fp:
            pickle.dump(data_list, fp)


def load_from_json(file_name):
    try:
        with open(file_name, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None


def load_doc_ids(index_dir):
    '''
    index_dir: index directory where document/segment ids file is stored
    '''

    ids_list_file = f"{index_dir}/seg_ids_list.json"
    ids_dict = load_from_json(ids_list_file)
    return ids_dict

def search_colbert(queries, passages_list, colbert_name, index_dir, logger, 
                  experiment='colbert-indexing', topk=200):
    
    queries = Queries(data=queries)
    collection = Collection(data=passages_list)
    logger.info(f'Loaded {len(queries)} queries and {len(collection):,} passages')

    with Run().context(RunConfig(nranks=1, experiment=experiment, overwrite=True)):
        searcher = Searcher(index=colbert_name, collection=collection, index_root=index_dir)
        ranking = searcher.search_all(queries, k=topk)

        save_path = f"{index_dir}/{colbert_name}/{colbert_name}_ranking_topk_{topk}.tsv"
        # if os.path.exists(save_path): # check and delete previous ranking file
        #     os.remove(save_path)
        ranking.save(save_path)
        logger.info(f"Done retreival. Ranking was saved to {save_path}")
        return ranking.todict()



def convert_ranking_to_trec_run(colbert_ranking, doc_idx_to_id, query_idx_to_id, run_path, tag='colbert'):

    output = ""
    for q_idx in colbert_ranking.keys():
        for doc_idx, rank, score in colbert_ranking[q_idx]:
            doc_idx = str(doc_idx)
            docno = doc_idx_to_id[doc_idx]
            qid = query_idx_to_id[q_idx]
            line = f"{str(qid)}\tQ0\t{docno}\t{rank}\t{score}\t{tag}"
            output += line + "\n"

    if not os.path.exists(os.path.dirname(run_path)):
        os.makedirs(os.path.dirname(run_path), exist_ok=True)
    with open(run_path, 'w') as file: 
        file.write(output)
    return output




def load_queries(queries_path):
    print("#> Loading the queries from", queries_path, "...")
    queries = OrderedDict()
    idx_to_id = {}

    idx = 1
    with open(queries_path) as f:
        for line in f:
            qid, query, *_ = line.strip().split('\t')
            idx_to_id[idx] = qid
            assert (qid not in queries), ("Query QID", qid, "is repeated!")
            queries[idx] = query
            idx += 1
            # if idx > 10:
            #     break
    return queries, idx_to_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", required=True, type=str, help="Path to a .tsv queries file without header")
    parser.add_argument("--corpus_path", required=True, type=str, help="Path to a .tsv corpus file without header")
    parser.add_argument("--colbert_name", required=True, type=str, help="Name of Colbert model")
    parser.add_argument("--index_dir", required=True, type=str, help="index directory")
    parser.add_argument("--run_path", required=True, type=str, help="index directory")
    parser.add_argument("--log_file", required=True, type=str, help="File to save the log")
    parser.add_argument("--topk", required=False, default=200, type=int, help="index batch size")

    args = parser.parse_args()

    query_path = args.query_path
    corpus_path = args.corpus_path
    run_path = args.run_path
    topk = args.topk
    colbert_name = args.colbert_name
    log_file = args.log_file
    index_dir = args.index_dir
    logger = get_logger(log_file=log_file)

    df_col = pd.read_csv(corpus_path, sep='\t', names=['id', 'text'],) # nrows=10000)
    queries, q_idx_to_id = load_queries(query_path)

    doc_ids_dict = load_doc_ids(f"{index_dir}/{colbert_name}") 
    passages_list = list(df_col['text'])

    ranking_dict = search_colbert(queries, passages_list, colbert_name, index_dir, logger, topk=topk)
    convert_ranking_to_trec_run(colbert_ranking=ranking_dict, doc_idx_to_id=doc_ids_dict, 
                                query_idx_to_id=q_idx_to_id, run_path=run_path, tag='colbert')



if __name__ == "__main__":
    main()
