# -*- coding: utf-8 -*-
"""Part of this file was copeid from this repository  https://github.com/trecpodcasts/podcast-audio-feature-extraction/blob/main/src/data.py"""

import json, glob
import os
import pandas as pd
import numpy as np
import os, json, logging
import pandas as pd
import xml.etree.ElementTree as ET
import traceback
import pickle
import gzip
import shutil


class TrecRun(object):
    def __init__(self, filename=None):
        if filename is not None:
            self.read_run(filename)
        else:
            self.filename = None
            self.run_data = None

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        if self.filename:
            return "Data from file %s" % (self.get_full_filename_path())
        else:
            return "Data file not set yet"

    def get_runid(self):
        return self.run_data["system"][0]

    def rename_runid(self, name):
        self.run_data["system"] = name

    def validate_run_format(self):
        # any empty values in other columns will results in a NaN in the last column
        # (i.e., `system` column).
        check_df = self.run_data["system"].isnull()
        if any(check_df):
            malformed_line = check_df.argmax()
            raise ValueError(f"Malformed line {malformed_line}.")

        # check for duplicate docnos for the the same qid
        check_df = self.run_data.duplicated(subset=["qid", "docno"])
        if any(check_df):
            malformed_line = check_df.argmax()
            raise ValueError(f"Duplicated docno in line {malformed_line}.")

    def read_run(self, filename, run_header=None):
        # Replace with default argument for run_header
        if run_header is None:
            run_header = ["qid", "Q0", "docno", "rank", "score", "system"]

        # Set filename
        self.filename = filename

        # Read data from file
        self.run_data = pd.read_csv(filename, sep="\s+", names=run_header)
        self.validate_run_format()

        # Enforce string type on docno column (if present)
        if "docno" in self.run_data:
            self.run_data["docno"] = self.run_data["docno"].astype(str)
        # Enforce string type on q0 column (if present)
        if "Q0" in self.run_data:
            self.run_data["Q0"] = self.run_data["Q0"].astype(str)
        # Enforce string type on qid column (if present)
        if "qid" in self.run_data:
            self.run_data["qid"] = self.run_data["qid"].astype(str)

        # ranks = list(self.run_data['rank'])
        # pred_ranks = list(range(1, len(self.run_data)+1))
        # for i in range(len(ranks)):
        #     if ranks[i] != pred_ranks[i]:
        #         print(f"index i = {i} rank[i] = {ranks[i]} != pred_rank[i] = {pred_ranks[i]}")
        # # if pred_ranks != ranks:
        # #     print(len(pred_ranks))
        # #     print(len(ranks))
        # assert ranks == pred_ranks

        len_before = len(self.run_data)
        # Make sure the values are correctly sorted by score
        # print(f"length of run {filename} before sorting is {len(self.run_data)}")
        # self.run_data.sort_values(["qid", "docno", "score"], inplace=True, ascending=[True, True, False])

        # Sort based on the float column while keeping (col1, col2) pairs together
        # self.run_data = self.run_data.groupby(['qid', 'docno'], sort=False).first().reset_index().sort_values(by='score', ascending=False)
        len_after = len(self.run_data)
        if len_before != len_after:
            print(f"error in sorting {len_before} != {len_after}")
        # print(f"length of run {filename} after sorting is {len(self.run_data)}")





def clean_string(text):
    '''
    to fix utf-8 text issues and convert to ascii
    '''
    from cleantext import clean
    cleaned = clean(text, fix_unicode=True, to_ascii=True, lower=False, no_line_breaks=True, lang='en')
    return cleaned


def read_run(run):
    return pd.read_csv(run, sep=r'\s+', names=["qid", "Q0", "docno", "rank", 'score', 'tag'], 
                       dtype={"qid": "str", "docno": "str"})


def read_podcast_run(run_file):
    """Removes the extra prefix from the documents ids and the extra suffix decimal point with the following zero"""
    df_run = read_run(run_file)
    df_run['docno'] = df_run['docno'].apply(lambda docno: docno.split(".")[0]) # remove the decimal point and the following zero
    df_run['docno'] = df_run['docno'].apply(lambda docno: docno.split("spotify:episode:")[1]) # remove this prefix 'spotify:episode:'
    return df_run


def parse_xml_to_df(xml_file):
    """Returns a dataframe after parsing the input xml file"""

    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for child in root:
        row = {}
        for subchild in child:
            row[subchild.tag] = subchild.text
        data.append(row)

    return pd.DataFrame(data)


def format_query_file_in_terrier(xml_query_file,
                            save_file,
                            save_columns=['qid', 'query']):
    """Parse the input xml file to dataframe and then format it in pyterrier format"""
    df = parse_xml_to_df(xml_query_file)
    df['qid'] = df['num']
    df[save_columns].to_csv(save_file, sep='\t', index=False,)
    return df[save_columns]


def preprocess_podcast_qrels(qrels, save_file):
    """Removes the extra prefix from the documents ids and the extra suffix decimal point with the following zero"""
    df_qrels = pd.read_csv(qrels, names=['qid', 'Q0', 'docno', 'label'], sep='\s+', dtype={"qid": "str", "docno": "str"})
    df_qrels['docno'] = df_qrels['docno'].apply(lambda docno: docno.split(".")[0]) # remove the decimal point and the following zero
    df_qrels['docno'] = df_qrels['docno'].apply(lambda docno: docno.split("spotify:episode:")[1]) # remove this prefix 'spotify:episode:'
    df_qrels.to_csv(save_file, sep='\t', index=False, header=False)
    return df_qrels





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


def load_metadata(dataset_path):
    """Load the Spotify podcast dataset metadata."""
    return pd.read_csv(os.path.join(dataset_path, "metadata.tsv"), delimiter="\t")


def relative_file_path(show_filename_prefix, episode_filename_prefix):
    """Return the relative filepath based on the episode metadata."""
    return os.path.join(
        show_filename_prefix[5].upper(),
        show_filename_prefix[6].upper(),
        show_filename_prefix,
        episode_filename_prefix,
    )


def find_paths(metadata, base_folder, file_extension):
    """Find the filepath based on the dataset structure.

    Uses the metadata, the filepath folder and the file extension you want.

    Args:
        metadata (df): The metadata of the files you want to create a path for
        base_folder (str): base directory for where data is (to be) stored
        file_extension (str): extension of the file in the path

    Returns:
        paths (list): list of paths (str) for all files in the given metadata
    """
    paths = []
    for i in range(len(metadata)):
        relative_path = relative_file_path(
            metadata.show_filename_prefix.iloc[i],
            metadata.episode_filename_prefix.iloc[i],
        )
        path = os.path.join(base_folder, relative_path + file_extension)
        paths.append(path)
    return paths


def load_transcript(path):
    """Load a python dictionary with the .json transcript."""
    with open(path, "r") as file:
        transcript = json.load(file)
    return transcript


def retrieve_full_transcript(transcript_json):
    """Load the full transcript without timestamps or speakertags."""
    transcript = ""
    for result in transcript_json["results"][:-1]:
        transcript += result["alternatives"][0]["transcript"]
    return transcript


def retrieve_timestamped_transcript(path, with_speakers=False):
    """Load the full transcript with timestamps."""
    with open(path, "r") as file:
        transcript = json.load(file)

    starts, ends, words, speakers = [], [], [], []
    for word in transcript["results"][-1]["alternatives"][0]["words"]:
        starts.append(float(word["startTime"].replace("s", "")))
        ends.append(float(word["endTime"].replace("s", "")))
        words.append(word["word"])
        if with_speakers:
            speakers.append(word["speakerTag"])

    starts = np.array(starts, dtype=np.float32)
    ends = np.array(ends, dtype=np.float32)
    words = np.array(words)
    if with_speakers:
        speakers = np.array(speakers, dtype=np.int32)
    return {"starts": starts, "ends": ends, "words": words, "speaker": speakers}



def split_file(input_file, output_prefix, num_parts):
    with open(input_file, 'r') as f:
        # Read all lines from the input file
        lines = f.readlines()
    
    # Calculate the number of lines per part
    lines_per_part = len(lines) // num_parts
    remainder = len(lines) % num_parts  # Handle the case when total lines is not exactly divisible by num_parts

    # Create the output directory if it doesn't exist
    os.makedirs(output_prefix, exist_ok=True)

    # Write the lines to separate output files
    for i in range(num_parts):
        start_index = i * lines_per_part
        end_index = (i + 1) * lines_per_part if i < num_parts - 1 else len(lines)
        if remainder > 0:
            end_index += 1
            remainder -= 1

        output_file = os.path.join(output_prefix, f"part_{i}.jsonl")
        with open(output_file, 'w') as f_out:
            f_out.writelines(lines[start_index:end_index])



def unzip_file(zipped_file, unzipped_file):
    try:
        command = f"gunzip -c {zipped_file} > {unzipped_file}"
        os.system(command)
        return True
    except Exception as e:
        print(traceback.format_exc())
        return False


def unzip_directory(directory = '/storage/users/watheq/projects/podcast_resource/data/trec_runs_2020/retrieval',
                    unzip_directory = '/storage/users/watheq/projects/podcast_resource/data/trec_runs_2020/retrieval/unzipped',
                    ):
    '''
    directory: the input directory that contain .gz files
    unzip_directory: the directory to store the unzipped files
    '''
    # Create the unzip directory if it doesn't exist
    if not os.path.exists(unzip_directory):
        os.makedirs(unzip_directory)

    # Iterate over all .gz files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.gz'):
            # Create the directory name for the unzipped file
            dir_name = os.path.splitext(filename)[0]
            # print(dir_name)
            # break
            # unzipped_dir = os.path.join(unzip_directory, dir_name)
            # if not os.path.exists(unzipped_dir):
            #     os.makedirs(unzipped_dir)

            # Unzip the file
            with gzip.open(os.path.join(directory, filename), 'rb') as gz_file:
                with open(os.path.join(unzip_directory, filename.replace('.gz', '')), 'wb') as out_file:
                    out_file.write(gz_file.read())

def get_run_file(runs_dir, run_name):
    zipped_run = os.path.join(runs_dir, f"{run_name}.res.gz")
    unzipped_run = os.path.join(runs_dir, f"{run_name}.run")
    unzip_file(zipped_file=zipped_run, unzipped_file=unzipped_run)
    return unzipped_run


def read_qrels(qrels_file):
    return pd.read_csv(qrels_file, names=['qid', 'Q0', 'docno', 'label'], sep='\s+', dtype={"qid": "str", "docno": "str"})


def read_jsonl(jsonl_file):
    return pd.read_json(jsonl_file, lines=True)


def read_query(query_file):
    return pd.read_csv(query_file, sep='\t', dtype={"qid": "str",})


def write_line(file, new_line, mode='a'):
    try:
        # Create the output directory if it doesn't exist
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file), exist_ok=True)

        with open(file, mode) as file:
            file.write(new_line)  
    except Exception as e:
        raise Exception("Writing error: {}".format(e))
    

    

def read_json(file_name):
    
    with open(file_name) as f:
        data = json.loads(f.read())
    return data



def save_dict(file_name, my_dict):
    '''
    save_file: path to save the dictionary with .json extension
    '''
    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        json.dump(my_dict, f)

def load_dict(file_name):
    loaded_dict = json.load(open(file_name, 'r'))
    return loaded_dict


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


def load_list(file_name, method='json'):
    try:
        if method == 'json':
            with open(file_name, 'r') as file:
                loaded_list =  json.load(file)
        else:
            with open(file_name, "rb") as fp:   # Unpickling
                loaded_list = pickle.load(fp)
                
    except FileNotFoundError:
        print(f"File {file_name} not found.")
        return None

    return loaded_list


def check_save_dir(save_file):
    # Create the output directory if it doesn't exist
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)

def find_files(directory, pattern):
    files = [file for file in glob.glob(directory + '/' + pattern)]
    # print(files)  # prints the list of files matching the pattern
    # if len(files) == 0:
    #     print(f"Found zero file matching the pattern")
    # else:
    #     print(f"Found {len(files)} files matching the pattern")
    return files