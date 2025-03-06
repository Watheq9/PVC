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


def index_colbert(passages_list, index_dir, logger, doc_maxlen, nbits, 
                  checkpoint, experiment='colbert-indexing', index_bsize=64):
    

    logger.info(f"Started indexing ... ")
    collection = Collection(data=passages_list)
    with Run().context(RunConfig(nranks=1, experiment=experiment)):  # nranks specifies the number of GPUs to use.
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits, index_bsize=index_bsize)
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_dir, collection=collection, overwrite=True)

    logger.info(f"Done indexing. The index was saved to {index_dir}")


def filter_null_segments(df_col, save_dir, logger):
    '''
    save_dir: directory to save ids of null and non-null segments
    '''

    ids_list_file = f"{save_dir}/seg_ids_list.json"
    ids_of_null_segments =f"{save_dir}/ids_of_null_segments.json"

    num_of_total_rows = len(df_col)
    # extract ids of null segments, save them to a file, and then exclude them from indexing
    null_rows_df = df_col[df_col['text'].isnull()]
    ids_of_null_rows = list(null_rows_df['id'])
    save_list(file_name=ids_of_null_segments, data_list=ids_of_null_rows)

    # keep non-values segments
    df_col = df_col[df_col['text'].notnull()]

    assert len(df_col) == num_of_total_rows - len(ids_of_null_rows)
    logger.info(f"Number of null segments is {len(ids_of_null_rows)}")
    logger.info(f"Number of non-null segments is {len(df_col)}")

    ids_dict = {}
    int_ids = []

    for idx, id in enumerate(df_col['id'].values):
        ids_dict[idx] = id
        int_ids.append(idx)
    df_col['int_id'] = int_ids
    save_dict(ids_list_file, ids_dict)
    return df_col



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path", required=True, type=str, help="Path to a .tsv corpus file without header")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to the colbert checkpoint")
    parser.add_argument("--nbits", required=True, type=int, help="Number of bits to encode each dimension")
    parser.add_argument("--doc_maxlen", required=True, type=int, help="Maximum length before applying truncation")
    parser.add_argument("--index_dir", required=True, type=str, help="Directory to save the index")
    parser.add_argument("--log_file", required=True, type=str, help="File to save the log")
    parser.add_argument("--index_bsize", required=False, default=64, type=int, help="index batch size")
    args = parser.parse_args()

    corpus_path = args.corpus_path
    checkpoint = args.checkpoint
    nbits = args.nbits
    doc_maxlen = args.doc_maxlen
    log_file = args.log_file
    index_dir = args.index_dir
    index_bsize = args.index_bsize
    logger = get_logger(log_file=log_file)

    df_col = pd.read_csv(corpus_path, sep='\t', names=['id', 'text'],) # nrows=10000)
    df_col = filter_null_segments(df_col, save_dir=index_dir, logger=logger)

    passages_list = list(df_col['text'])

    index_colbert(passages_list, index_dir, logger, doc_maxlen, nbits, 
                  checkpoint, experiment='colbert-indexing', index_bsize=index_bsize)



if __name__ == "__main__":
    main()


'''
CUDA_VISIBLE_DEVICES=${GPU_NUMBER} python systems/colBERT/index.py \
--query_path \
--corpus_path \
--checkpoint \
--nbits \
--doc_maxlen \
--index_dir \
--log_file 

'''