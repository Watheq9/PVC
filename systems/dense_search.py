


from tqdm import tqdm
import pandas as pd
import os, sys
sys.path.insert(0, os.getcwd())
from FlagEmbedding import FlagModel
import numpy as np
import faiss
import math
import csv
import os
import pickle
import time
import helper.utils as myutils
import faiss
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from FlagEmbedding import BGEM3FlagModel
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_embeddings(embedding_cache_path):
    print("Load pre-computed embeddings from disc")
    doc_ids = None
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data["sentences"]
        corpus_embeddings = cache_data["embeddings"]
        if "doc_ids" in cache_data:
            doc_ids = cache_data["doc_ids"]
        print(type(corpus_embeddings))
        corpus_embeddings = torch.tensor(corpus_embeddings, device=device)
        print(type(corpus_embeddings))
        return corpus_sentences, corpus_embeddings, doc_ids
    

def load_faiss_index(path):
    index = faiss.read_index(path)
    return index



def brute_force_search(query_file, model, doc_ids, run_save_file, corpus_sentences=None,
           topk=1000, corpus_embeddings=None, post_processing=False, max_length=768, model_name=""):

    df_queries = pd.read_csv(query_file, sep='\t', dtype={"qid": "str", "query": "str"}, names=['qid', 'query'])
    queries = np.array(df_queries["query"])
    q_ids = np.array(df_queries["qid"])
    for i in tqdm(range(0, len(queries)), desc="Searching"):

        if 'bge-m3' in model_name: 
            query_embedding = model.encode(queries[i], max_length=max_length)['dense_vecs'] 
        else:
            query_embedding = model.encode(queries[i], max_length=max_length, convert_to_numpy=True,)
    

        # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
        # Here, we use semantic search to search without quantization or any compression
        num_retrieved = int(math.ceil(topk*1.3)) # retrieve topk*1.3 to compensate removing the short segments if any
        hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=num_retrieved)[0]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)

        # hits_ids = set([hit["corpus_id"] for hit in hits])
        hit_idx = 0
        while hit_idx < topk:
            rank = hit_idx + 1
            corpus_id = int(hits[hit_idx]['corpus_id'])
            sentence = corpus_sentences[corpus_id]
            if post_processing and len(sentence.split()) < 2: # don't add the segment if it is very small
                hit_idx += 1
                continue
            
            line = f"{q_ids[i]}\tQ0\t{doc_ids[corpus_id]}\t{rank}\t{hits[hit_idx]['score']}\t{model_name}"
            # print(line)
            # print("\t{:.3f}\t{}".format(hits[hit_idx]["score"], sentence))

            hit_idx += 1
            try:
                with open(run_save_file, 'a') as file:
                    file.write(line + '\n')  # Write the JSON line to the file
            except Exception as e:
                raise Exception("Writing error: {}".format(e))




def search(query_file, model, doc_ids, index, run_save_file,
           topk=1000, corpus_sentences=None, post_processing=False, max_length=768, model_name=""):

    df_queries = pd.read_csv(query_file, sep='\t', dtype={"qid": "str", "query": "str"})
    queries = np.array(df_queries["query"])
    q_ids = np.array(df_queries["qid"])
    for i in tqdm(range(0, len(queries)), desc="Searching"):

        if 'bge-m3' in model_name: 
            query_embedding = model.encode(queries[i], max_length=max_length, device=device)['dense_vecs'] 
        else:
            query_embedding = model.encode(queries[i], max_length=max_length, convert_to_numpy=True, device=device)
    
        # FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = np.expand_dims(query_embedding, axis=0)

        distances, corpus_ids = index.search(query_embedding, topk*2) # retrieve top*k to compensate removing the short segments if any

        # We extract corpus ids and scores for the first query
        hits = [{"corpus_id": id, "score": score} for id, score in zip(corpus_ids[0], distances[0])]
        hits = sorted(hits, key=lambda x: x["score"], reverse=True)

        print("Input question:", queries[i])
        hit_idx = 0
        while hit_idx < topk:
            rank = hit_idx + 1
            corpus_id = int(hits[hit_idx]['corpus_id'])
            sentence = corpus_sentences[corpus_id]
            if post_processing and len(sentence.split()) < 5: # don't add the segment if it is very small
                hit_idx += 1
                continue
            
            line = f"{q_ids[i]}\tQ0\t{doc_ids[corpus_id]}\t{rank}\t{hits[hit_idx]['score']}\tDense"
            # print(line)
            # print("\t{:.3f}\t{}".format(hits[hit_idx]["score"], sentence))

            hit_idx += 1
            try:
                with open(run_save_file, 'a') as file:
                    file.write(line + '\n')  # Write the JSON line to the file
            except Exception as e:
                raise Exception("Writing error: {}".format(e))

def load_model(model_name, max_length = 768 ):
    if 'BAAI' in model_name:
        if 'bge-m3' in model_name: 
            model = BGEM3FlagModel(model_name, use_fp16=True, ) # use only one device, defaults use all available devices
        else:
            model = FlagModel(model_name,
                            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
                            use_fp16=True,)
    else:
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device).cuda()
        model.max_seq_length = max_length   

    # model = SentenceTransformer(model_name, trust_remote_code=True).cuda()
    # model.max_seq_length = max_length   
    return model


def main():
    

    parser = argparse.ArgumentParser(description="Script to build faiss index using an embedding file")
    parser.add_argument("--index_path", type=str, required=False, default=None, help="Path to the index file ")
    parser.add_argument("--model_name", type=str, required=True, help="The encoding model")
    parser.add_argument("--query_file", type=str, required=True, help="Path to the query file ")
    parser.add_argument("--embed_file", type=str, required=True, help="Path to the corpus embedding file ")
    parser.add_argument("--topk", type=int, required=True, help="how many documents to retrieve")
    parser.add_argument("--log", type=str, required=True, help="Path to save the execution log")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the run file")
    parser.add_argument("--apply_post_processing",action="store_true",help="If True, remove segments with a few words in content",)
    parser.add_argument("--nprobe", type=int, required=False, default=20, help="Number of clusters to explorer at search time.")
    parser.add_argument("--search_type", type=str, required=False, default="quantized", help="Can be falt or quantized. Flat applies the brute force semantic search, quantized: uses faiss index search.")

    # parser.add_argument("--embedding_size", type=int, required=True, help="The embedding size of the encoding model")
    args = parser.parse_args()
    index_path = args.index_path
    model_name = args.model_name
    query_file = args.query_file
    embed_file = args.embed_file
    topk = int(args.topk)
    log_file = args.log
    output = args.output
    apply_post_processing = args.apply_post_processing
    search_type = args.search_type
    nprobe = int(args.nprobe)

    logger = myutils.get_logger(log_file)
    logger.info(f"The log_file will be saved to {log_file}")
    logger.info(f"The output file will be saved to {output}")

    model =  load_model(model_name)


    corpus_sentences, corpus_embeddings, seg_ids = load_embeddings(embed_file)
    logger.info(f"Loaded embeddings from {embed_file} {os.path.isfile(embed_file)}")

    with open(output, 'w') as file: # Write to new file (clear any content)
        file.write("")
    
    if search_type == "flat":
        logger.info(f"Applying brute force (flat) semantic search (unquantized) using the model {model_name} for encoding queries")
        brute_force_search(query_file, model, doc_ids=seg_ids, run_save_file=output, corpus_sentences=corpus_sentences,
           topk=topk, corpus_embeddings=corpus_embeddings, post_processing=apply_post_processing, model_name=model_name)
    else:
        logger.info(f"The input index is read from {index_path}") #  and embedding size is {embedding_size}")
        logger.info(f"Applying quantized search using the model {model_name} for encoding queries")
        index = load_faiss_index(path=index_path)
        index.nprobe = nprobe
        logger.info(f"index was loaded from {index_path}")
        search(query_file, model, seg_ids, index, run_save_file=output,
            topk=topk, corpus_sentences=corpus_sentences, post_processing=apply_post_processing)



if __name__ == "__main__":
    main()





