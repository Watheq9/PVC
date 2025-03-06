# -*- coding: utf-8 -*-

"""Script to index podcast segments to Elasticsearch."""

import os, sys
sys.path.insert(0, os.getcwd())
import math
import configure as conf
import numpy as np
from tqdm import tqdm
from helper import utils
# import utils
import json, re
import argparse
import traceback


class PodcastSegment():
    """Implementation of a podcast segment"""


    def __init__(self, id, show_name, show_desc, epis_name,
                epis_desc, seg_words, minutes_ids=-1,):
        self.id = id
        self.seg_words = seg_words
        self.show_name = show_name
        self.show_desc = show_desc
        self.epis_name = epis_name
        self.epis_desc = epis_desc
        self.minutes_ids = minutes_ids
        
    def save(self, filename):
        """Save the segment to a json file."""
        self.seg_length = len(self.seg_words.split())
        with open(filename, 'w') as file:
            json.dump(self.__dict__, file)
        # json_dict = json.dumps(self.__dict__)
        return True
    
    def get_dict(self):
        return self.__dict__


def clean_text(text):
    """Clean the text to remove non-topical content.

    This includes things like episode numbers, advertisements, and links.
    """

    def isNaN(string):
        return string != string

    # For now just check it is not NaN
    if isNaN(text):
        text = ""

    text = re.sub(r"http\S+", " ", text)  # remove urls
    text = re.sub(r"RT ", " ", text)  # remove rt
    text = re.sub(r"@[\w]*", " ", text)  # remove handles
    text = re.sub(r"[\.\,\#_\|\:\?\?\/\=]", " ", text) # remove special characters
    text = re.sub(r"\t", " ", text)  # remove tabs
    text = re.sub(r"\n", " ", text)  # remove line jump
    text = re.sub(r"\s+", " ", text)  # remove extra white space
    text = text.strip()
    return text



def get_minutes_ids(seg_base, start_time, end_time, step=60):
    m_start = math.floor(start_time / step) * step
    m_end = math.floor(end_time / step) * step
    ids = []
    seg_min = m_start
    while seg_min <= m_end:
        ids.append(seg_base +str(seg_min))
        seg_min += step
    return ids


def form_time_segment(transcript, seg_base, seg_length, start_time, extra_sec=0):
    '''
    extra_sec: extra seconds to add before and after each segment, eg., 5 adds 5 seconds before the segment and 5 seconds after the segments
    '''

    last_word_time = math.ceil(transcript["starts"][-1]) # last_word_time
    end_time = min(start_time + seg_length - 0.01, last_word_time) # - 0.01 to avoid the overlap with the next segment

    segment_id = seg_base + str(start_time)

    if extra_sec != 0:
        start_time = max(0, start_time - extra_sec)
        end_time = min(end_time + extra_sec, last_word_time)

    # Find the words in the segment
    word_indices = np.where(np.logical_and(
                                    transcript["starts"] >= start_time,
                                    transcript["starts"] <= end_time))[0]
    segment = transcript["words"][word_indices]
    segment = " ".join(segment)

    # get the ids of minutes in this segment
    minutes_ids = get_minutes_ids(seg_base, start_time, end_time)

    return segment_id, segment, minutes_ids


def form_words_segment(transcript, seg_base, seg_length, start_word_index):

    last_word_time = math.ceil(transcript["starts"][-1]) # last_word_time
    num_of_words = len(transcript['words'])
    end_word_index = min(start_word_index + seg_length, num_of_words) - 1
    seg_range = end_word_index - start_word_index + 1
    start_word_time = transcript['starts'][start_word_index]
    end_word_time = transcript['starts'][end_word_index]

    segment_id = seg_base + str(start_word_time) +"_" + str(end_word_time)
    segment = ""

    # form the segment 
    for i in range(seg_range):
        segment += transcript['words'][start_word_index + i] + " "

    # get the ids of minutes in this segment
    minutes_ids = get_minutes_ids(seg_base, start_word_time, end_word_time)

    return segment_id, segment, minutes_ids



def get_silero_transcript(input_file,):
    with open(input_file, "r") as file:
        transcript = json.load(file)
    # (-1) last element in the transcript is the full transcript
    # (-2) before last element is the last word in the transcript 
    words = np.array(list(map(lambda d: d['word'], transcript[:-1]))) # without the full transcript
    starts = np.array(list(map(lambda d: d['start'], transcript[:-1])), dtype=np.float32)
    ends = np.array(list(map(lambda d: d['end'], transcript[:-1])), dtype=np.float32)
    return {"starts": starts, "ends": ends, "words": words}




def get_whisperX_transcript(input_file, logger=None):
    with open(input_file, "r") as file:
        transcript = json.load(file)

    transcript = transcript['word_segments'] # this contains all timestamped words of the transcribed audio
    if len(transcript) <= 1:
        logger.info(f"Empty transcript in {input_file}. Here is the transcript: {transcript}")
        return {}

    words = []
    starts = []
    ends = []
    for i, line in enumerate(transcript):

        word = line['word']
        if 'start' in line:
            start = line['start']
            end = line['end']
        # if there is no 'start' key, this means it is a number
        # And the numbers don't have timestamps in whisperX, so we add them manually as the average time between the previous and next words
        elif i == 0: # at the begining 
            j = i +1
            while 'start' not in transcript[j]: # until you find timestamped word
                j += 1
            start = max(0.0, transcript[j]['start'] - 0.7 * (j - i))
            end = start + 0.7
            # print(f"start = {start}, end = {end}")
            # print(transcript[:j+1])
        elif i < len(transcript) : # anything before the end
            start = ends[-1] 
            end = start + 0.7
        # else: # between two words
        #     logger.info(f" i = {i} transcript[i-1] = {transcript[i-1]} \n transcript[i+1] = {transcript[i+1]}")
        #     start = (transcript[i-1]['start'] + transcript[i+1]['start'])/ 2
        #     end = (transcript[i-1]['end'] + transcript[i+1]['end'])/ 2
        
        words.append(word)
        starts.append(start)
        ends.append(end)

    # print(starts[:7])
    words = np.array(words) # without the full transcript
    starts = np.array(starts, dtype=np.float32)
    ends = np.array(ends, dtype=np.float32)

    return {"starts": starts, "ends": ends, "words": words}


def add_podcast(
        transcript_path,
        show_name,
        show_desc,
        epis_name,
        epis_desc,
        seg_length=120,
        seg_step=60,
        segment_type='time',
        output_file="",
        model="",
        extra_sec=0,
        logger=None,
    ):
    """Save podcast transcript to jsonl file.
    segment_type: 'time' means form segment based on time by selecting the words within the timeslot specified by seg_length and the next segment 
                        has to start after making time jump set by seg_step
                  'words' means form segment based on number of words by adding words until reaching the length specified by seg_length and the next 
                  segment will start after seg_step words.
    """
    # Generate the segment basename
    seg_base = os.path.splitext(os.path.basename(transcript_path))[0] + "_" # seg_base is actually the episode_filename_prefix

    # Clean the show and episode names and descriptions
    show_name = clean_text(show_name)
    show_desc = clean_text(show_desc)
    epis_name = clean_text(epis_name)
    epis_desc = clean_text(epis_desc)

    # Get the transcript and make it as dictionary of starts, ends, and words
    if model.startswith('silero'):
        transcript = get_silero_transcript(transcript_path)
    elif model.startswith('whisperX'):
        transcript = get_whisperX_transcript(transcript_path, logger)
    else: # Google transcript (the default)
        transcript = utils.retrieve_timestamped_transcript(transcript_path)
    
    # if len(transcript) == 0:
    #     return 
    
    start = 0
    if segment_type == 'time':
        end = math.ceil(transcript["starts"][-1]) # last_word_time
    else: # words segment
        end = len(transcript['words']) # number of words in the transcript

    # Generate the segments from the start to the end of the podcast
    while start < end:  

        if segment_type == 'time':
            segment_id, segment, minutes_ids = form_time_segment(transcript, seg_base, seg_length, start_time=start, extra_sec=extra_sec)
        else:
            segment_id, segment, minutes_ids = form_words_segment(transcript, seg_base, seg_length, start_word_index=start)

        # Create and save the segment
        segment = PodcastSegment(
                        id=segment_id,
                        show_name=show_name,
                        show_desc=show_desc,
                        epis_name=epis_name,
                        epis_desc=epis_desc,
                        seg_words=segment,
                        minutes_ids=minutes_ids)
        
        # set the start of next segment
        start = start + seg_step
        
        try:
            with open(output_file, 'a') as file:
                file.write(json.dumps(segment.get_dict()) + '\n')  # Write the JSON line to the file
        except Exception as e:
            raise Exception("Writing error: {}".format(e))



def main():
    parser = argparse.ArgumentParser(description="Script to generate podcast segments")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file")
    parser.add_argument("--save_file", type=str, required=True, help="Path to save the resulted segments")
    parser.add_argument("--seg_length", type=str, required=True, help="Length of each segment")
    parser.add_argument("--seg_step", type=str, required=True, help="The step to jump when moving to next segment")
    parser.add_argument("--seg_type", type=str, default='time', required=True, help="The segment type which can be either 'time' or 'words'")
    parser.add_argument("--model", type=str, default='Google', required=False, help="The model used in generating the transcript")
    parser.add_argument("--extra_sec", type=int, default=0, required=False, help="Extra seconds to add on both ends of each each segmen")
    parser.add_argument("--transcript_dir", type=str, default=conf.TREC2020_dataset, required=False, help="Directory to corpus transcripts")
    args = parser.parse_args()
    log_file = args.log_file
    save_file = args.save_file
    seg_length = int(args.seg_length)
    seg_step = int(args.seg_step)
    seg_type = args.seg_type
    model = args.model
    extra_sec = args.extra_sec
    transcript_dir = args.transcript_dir
    google_transcripts_path = os.path.join(conf.TREC2020_dataset, "podcasts-transcripts") # default transcripts directory 
    # log_file = '/storage/users/watheq/projects/podcast_search/data/logs/preparing_segments.txt'
    # segments_file = '/storage/users/watheq/projects/podcast_search/data/dataset_files/podcast_2M_segments-2.jsonl'

    logger = utils.get_logger(log_file)
    logger.info(f"The segment length = {seg_length}, segment_step is {seg_step} and the segement type is {seg_type}")
    logger.info(f"using transcript generated by {model} and loaded from {transcript_dir}")
    logger.info(f"The file will be saved to {save_file}")

    metadata = utils.load_metadata(conf.TREC2020_dataset)

    errorful_episodes = []
    no_transcripts = []

    with open(save_file, 'w') as file: # Write to new file (clear any content)
        file.write("")

    for index, row in tqdm(metadata.iterrows()):
        if model != "Google": # using different model
            transcript_file = os.path.join(transcript_dir, row["episode_filename_prefix"] + ".json")

        else: # using provided Google transcripts
            relative_file_path = utils.relative_file_path(row["show_filename_prefix"], row["episode_filename_prefix"])+ ".json"
            transcript_file = os.path.join(google_transcripts_path, relative_file_path)

        if not os.path.isfile(transcript_file):
            logger.info(f"no transcripts found for {transcript_file}")
            no_transcripts.append(transcript_file)
            continue
        try:
            add_podcast(
                    transcript_file,
                    row["show_name"],
                    row["show_description"],
                    row["episode_name"],
                    row["episode_description"],
                    seg_length=seg_length,
                    seg_step=seg_step,
                    segment_type=seg_type,
                    output_file=save_file,
                    model=model,
                    extra_sec=extra_sec, 
                    logger=logger)
        except Exception as e:
            logger.error(f'Could not add the podcast with  show_filename_prefix: {row["show_filename_prefix"]} and  episode_filename_prefix {row["episode_filename_prefix"]}. Exception details= {format(e)}')
            logger.info(traceback.format_exc())
            errorful_episodes.append(transcript_file)

    logger.info(f"Number of not transcriped episodes {len(no_transcripts)} and here they are {no_transcripts}")
    logger.info(f"Number of errorful epsiodes {len(errorful_episodes)} and here they are {errorful_episodes}")

if __name__ == "__main__":
    main()

