import os, sys
sys.path.insert(0, os.getcwd())
import logging
import pandas as pd
import re
import pyterrier as pt
from pyterrier_pisa import PisaIndex
import numpy as np
import argparse
import traceback
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer # Snowball Stemmer is also known as the Porter2
# nltk.download('stopwords')
# nltk.download('punkt')
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"], version='5.10')
# pt.java.add_package('com.github.terrierteam', 'terrier-prf', '-SNAPSHOT')
# pt.terrier.set_version('5.10')
# pt.java.init() # optional, forces java initialisation
from pyterrier import autoclass



porter = PorterStemmer()
porter2 = SnowballStemmer(language='english')  # porter2 is the default in Pisa Indexer
terrier_stopwords = autoclass("org.terrier.terms.Stopwords")(None) # terrier_stopwords is the default in Pisa Indexer 


def initailize_logger(logger, log_file, level):
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


def en_stem(text, stemmer=porter2):
    # # Tokenize the input string into words
    # words = nltk.word_tokenize(text)
    # # Apply stemming to each word
    # stemmed_words = [porter.stem(word) for word in words]
    # # Join the stemmed words back into a string
    # stemmed_string = " ".join(stemmed_words)

    token_words= word_tokenize(text)
    return " ".join([stemmer.stem(word) for word in token_words])


def lower_case(text):
    # apply preprocessing steps on the given sentence
    text = text.lower()
    return text

def clean(text):
    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"RT ", " ", text)  # remove rt
    text = re.sub(r"@[\w]*", " ", text)  # remove handles
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text) # remove special characters
    text = re.sub(r"\t", " ", text)  # remove tabs
    text = re.sub(r"\n", " ", text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space
    text = text.strip()
    return text


def remove_punctuation(text):
    # Removing punctuations in string using regex
    text = re.sub(r'[^\w\s]', ' ', text)
    return text


def remove_stop_words(text, apply_nltk=False):
    '''
    text: input text to remove stopwords from
    apply_nltk: False: means apply terrier_stopwords
                        True: means apply nltk stopwords
    '''
    if apply_nltk:
        stop_words = stopwords.words()
        text = " ".join(word for word in text.split() if word not in stop_words)
    else: # apply terrier stopwords
        text = " ".join(word for word in text.split() if not terrier_stopwords.isStopword(word))
    return text


def preprocess(text, stop_words='terrier'):
    text  = remove_punctuation(text)
    text  = clean(text)
    text = lower_case(text)
    if stop_words == 'terrier':
        text = remove_stop_words(text, apply_nltk=False)
    elif stop_words == 'nltk':
        text = remove_stop_words(text, apply_nltk=True)
    text = en_stem(text)
    return text



def load_corpus(collection_path):
    df_col = pd.read_json(collection_path, lines=True,)
    if 'docno' not in df_col.columns and 'id' in df_col.columns:
        df_col['docno'] = df_col['id'].astype('str')
    return df_col

def remove_short_segments(df, segment_column, logger):
    df['seg_len'] = df[segment_column].apply(lambda text: len(text.split())) # compute segment lengths
    df = df[df['seg_len'] > 5] # removed (~ 19725) segments where their lengths are less than 6 words
    logger.info("Done removing the segments with lenth less than 6 words. Perform indexing .....")
    return df

def get_document(df, fields):
    for i, row in df.iterrows():
        new_dict = {"docno": row["id"]}
        for field in fields:
            new_dict.update({field: row[field]})
        yield new_dict

def join_preprocess_columns(df, fields, meta_fields=[],):
    # Join lists in columns that contain them (predicted_queries)
    for col in fields:
        if df[col].apply(isinstance, args=(list,)).all(): # if there is a list of values in a column cell
            df[col] = df[col].apply(lambda x: ' \n '.join(map(str, x)))

    # Join lists in columns that contain them 
    for col in meta_fields:
        if df[col].apply(isinstance, args=(list,)).all():
            df[col] = df[col].apply(lambda x: ' '.join(map(str, x)))

    df['text'] = df[fields].apply(lambda x: ' \n '.join(x.astype(str)), axis=1)
    df['text'] = df['text'].apply(preprocess)

    # Drop the original columns that were merged
    # df.drop(columns=fields, inplace=True)
    return df 

def append_to_segments(df, append_fields):
    if len(append_fields) == 0:
        return df
    

def build_terrier_index(df, index_path, fields=[], logger=None, index_type='one-field', 
                        meta_fields=[],):

    # initialize the index
    iter_indexer = pt.IterDictIndexer(index_path, overwrite=True, verbose=True)
    iter_indexer.setProperty("tokeniser", "EnglishTokeniser")
    meta_fields.append('docno')


    if index_type == 'one-field':
        df = join_preprocess_columns(df, fields, meta_fields)
        indexref = iter_indexer.index(df.to_dict(orient="records"), 
                                      fields=['docno', 'text'], meta=meta_fields)

    elif index_type == 'multi-field':
        fields.append("docno")
        indexref = iter_indexer.index(df.to_dict(orient="records"), 
                                      fields=fields, meta=meta_fields)

    logger.info(f"Done indexing. Index is saved to {index_path}")
    return indexref


def build_pisa_index(df, index_path, fields=[], logger=None, segment_column='segment',):

    df = join_preprocess_columns(df, fields)
    # build the index on the two fields (docno & text)
    pisa_index = PisaIndex(index_path, stops='none', overwrite=True)
    index_ref = pisa_index.index(df[['docno', 'text']].to_dict(orient="records"))
    return index_ref


def build_index(index_type, corpus, index_dir, fields, logger, meta_fields=[],):
    try:
        df = load_corpus(corpus)
        logger.info("Done loading the whole corpus.")
        if index_type == 'terrier-one-field':
            build_terrier_index(df=df, index_path=index_dir, fields=fields, logger=logger, 
                                index_type='one-field', meta_fields=meta_fields, )
        elif index_type == 'terrier-multi-field':
            build_terrier_index(df=df, index_path=index_dir, fields=fields, logger=logger, 
                                index_type='multi-field', meta_fields=meta_fields,)
        elif index_type == 'pisa':
            build_pisa_index(df=df, index_path=index_dir, fields=fields, logger=logger,)
        else:
            raise Exception("Invalid value for the index_type argument" )
    except Exception as e:
        logger.error(f'Could not index due to this exception {format(e)}')
        logger.info(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(description="Script to build multi field index")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file")
    parser.add_argument("--corpus", type=str, required=True, help="Path to read the corpus from")
    parser.add_argument("--index_dir", type=str, required=True, help="Dirctory to save the index at")
    parser.add_argument("--index_type", type=str, required=True, help="Can be either pisa, terrier-one-field, or terrier-multi-field")
    parser.add_argument("--fields",  default=['seg_words'], required=False, metavar='string', nargs='+', help='List of fields to index')
    parser.add_argument("--meta_fields",  default=['docno'], required=False, metavar='string', nargs='+', help='List of fields to index as meta')
    args = parser.parse_args()
    log_file = args.log_file
    corpus = args.corpus
    index_type = args.index_type
    index_dir = args.index_dir
    fields = args.fields
    meta_fields = args.meta_fields


    logger = get_logger(log_file)
    logger.info(f"The input is read from {corpus}")
    logger.info(f"The index will be saved to {index_dir}")
    logger.info(f"These fields: {fields} will be indexed")
    logger.info(f"The index type is {index_type}")
    build_index(index_type, corpus, index_dir, fields, logger, meta_fields,)


if __name__ == "__main__":
    main()
