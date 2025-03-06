import os, sys
sys.path.insert(0, os.getcwd())
import re
import pandas as pd
import pyterrier as pt
import argparse
from pyterrier.measures import *
from pyterrier_pisa import PisaIndex
import traceback
from helper import indexing

def load_index(index_path):
    if isinstance(index_path, PisaIndex):
        index = PisaIndex(index_path, stops='none')
    else:
        index = pt.IndexFactory.of(index_path)
        print(index.getCollectionStatistics().toString())
    
    return index


def unzip_file(zipped_file, unzipped_file):
    try:
        command = f"gunzip -c {zipped_file} > {unzipped_file}"
        os.system(command)
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False


def get_run_file(runs_dir, run_name):
    zipped_run = os.path.join(runs_dir, f"{run_name}.res.gz")
    unzipped_run = os.path.join(runs_dir, f"{run_name}.run")
    unzip_file(zipped_file=zipped_run, unzipped_file=unzipped_run)
    return unzipped_run


def run_retrieval(index_path, run_path, query_file=None, retrieval_model='BM25', depth=1000, apply_preprocessing=True):
    
    if not os.path.exists(os.path.dirname(run_path)):
        os.makedirs(os.path.dirname(run_path))

    index = load_index(index_path)
    df_query = pd.read_csv(query_file, sep='\t', names=['qid', 'query'],
                             dtype={"qid": "str", "query": "str"},)# nrows=50)
    if apply_preprocessing:
        df_query['query'] = df_query['query'].apply(indexing.preprocess)

    retriever = pt.BatchRetrieve(index, wmodel=retrieval_model, verbose=True, num_results=depth, ) # controls={"bm25.b" : 0.4, "bm25.k_1": 0.9,}
    rm3_pipe = retriever >> pt.rewrite.RM3(index, fb_terms=10, fb_docs=3,) >> retriever
    results = rm3_pipe.transform(df_query)

    pt.io.write_results(results, run_path, run_name=retrieval_model)
    # results.to_csv(run_path, index=False)
    
    return results




def main():
    parser = argparse.ArgumentParser(description="Script to run lexical retrieval on a pyterrier index")
    parser.add_argument("--index_path", type=str, required=True, help="index path")
    parser.add_argument("--run_path", type=str, required=True, help="Path to save the resulted run")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file")
    parser.add_argument("--depth", type=int, required=False, default=1000, help="Path to save the resulted run")
    parser.add_argument("--model", type=str, required=False, default="BM25", help="Retrieval model to save the resulted run")
    args = parser.parse_args()
    index_path = args.index_path
    query_file = args.query_file
    run_path = args.run_path
    depth = args.depth
    retr_model = args.model
    retr_model = retr_model.split('+')[0]

    try:
        run_retrieval(index_path=index_path, run_path=run_path, depth=depth, retrieval_model=retr_model, query_file=query_file)
    except Exception as e:
        print(f'Could not continue due to this exception {format(e)}')




if __name__ == "__main__":
    main()