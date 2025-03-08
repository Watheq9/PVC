import os, sys
sys.path.insert(0, os.getcwd())
import toml, json
from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
import argparse
import helper.utils as utils
import tiktoken
import time


class JudgeResponse(BaseModel):
    relevance: int
    reason: str

    
class GPTJudge():
    def __init__(
        self,
        model_name="",
    ) -> None:
        self.model_name = model_name
        self.create_openai_client()
        self.temperature = 0
        self.top_p=1
        self.frequency_penalty=0.5
        self.presence_penalty=0
        self.system_prompt="You are a search quality rater evaluating the relevance of a given segment and query."
                

    def create_openai_client(self):
        api_key = os.environ["OPENAI_API_KEY"]
        # print(api_key)
        self.client = OpenAI(api_key=api_key)
        self.use_azure_ai = False

    
    def num_tokens_from_string(self, string: str, model: str) -> int:
        """Returns the number of tokens in a text string."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def get_JSON_response(self, messages, max_new_tokens, logger):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "llm_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "relevance": {"type": "integer"},
                                "reason": {"type": "string"}
                            },
                            "required": ["relevance", "reason"],
                            "additionalProperties": False
                        }
                    }
                },
                # strict=True
                )
            
            if response.choices[0].message.content:
                output = response.choices[0].message.content
            else:
                output = ""
                logger.error(f"Error in response, Empty outuput is {response.choices[0].message.content}")

        except Exception as e:
            if logger is None:
                print(f"Encountered {e} ")
            else:
                logger.error(f"Encountered {e} ")
            output = ""

        return output


    def get_object_response(self, messages, max_new_tokens, logger):
        try:
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                response_format=JudgeResponse,
                max_tokens=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
            response = completion.choices[0].message
            if response.parsed:
                output = response.parsed
            elif response.refusal:
                # handle refusal
                output = ""
                logger.error(f"Error in response, refusal, outuput is {response.refusal}")

        except Exception as e:
            # if type(e) == openai.LengthFinishReasonError:
            #     # Retry with a higher max tokens
            #     logger.error("Too many tokens: ", e)
            if logger is None:
                print(f"Encountered {e} ")
            else:
                logger.error(f"Encountered {e} ")

            output = ""
        
        return output
    

    def run_gpt(self, user_prompt, max_new_tokens, logger=None, system_prompt="", output_mode="JSON"):
        '''
        output_mode: can be:
            "JSON": JSON Schema 
            "object": The output is an object of the specified class
        '''
        if system_prompt == "":
            system_prompt = self.system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        if output_mode == "JSON":
            output = self.get_JSON_response(messages, max_new_tokens, logger)
        elif output_mode == "object":
            output = self.get_object_response(messages, max_new_tokens, logger)

        return output
    

    def get_system_prompt(self):
        return self.system_prompt
    




def read_missing(file):
    dic = ''
    with open(file,'r') as f:
        for i in f.readlines():
            dic=i #string
    dic = eval(dic) 
    return dic


def prepare_examples(random_pairs_input, joined_pairs_save_file, qrels_file):
    df_qrels = utils.read_qrels(qrels_file)
    df = utils.read_jsonl(random_pairs_input)

    labels = []

    mystr = ""
    for row in df.itertuples():
        qid = str(row.qid)
        docno = str(row.docno)
        relevance = df_qrels[(df_qrels['qid'] == qid ) & (df_qrels['docno'] == docno)]['label'].values[0]
        labels.append(relevance)

        mystr += f'''  For the Query: << {row.q_text} >>  with Query description: << {row.q_description} >> \n  and the Segment: << {row.doc_text} >>
            \n the relevance score is: {relevance} \n\n\n '''

    df['label'] = labels
    df.to_json(random_pairs_input, index=False, orient='records', lines=True)
    # print(mystr)
    mystr = json.dumps(mystr)
    utils.write_line(joined_pairs_save_file, mystr, mode='w')
    return mystr





def copy_judged_pairs_from_pool(pool_path, df_pairs, result, logger):
    judged_pairs_dict = {}
    pool_dict = {}
    df_pool = pd.read_json(pool_path, lines=True)
    for row in df_pool.itertuples(index=False):
        pool_dict[(str(row.qid), str(row.docno))] = row._asdict()
    cnt = 0
    for row in df_pairs.itertuples(index=False):            
        qid = str(row.qid)
        docno = str(row.docno)
        if (qid, docno) in pool_dict: 
            # already judged row, copy it to the result file
            line = json.dumps(pool_dict[(qid, docno)]) + '\n'
            utils.write_line(result, line, mode='a')
            judged_pairs_dict.update({(qid, docno): pool_dict[(qid, docno)]})
            cnt += 1
    
    logger.info(f"Copied {cnt} already judged pairs from {pool_path} to the result file at {result}")
    return judged_pairs_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Name of LLM model")
    parser.add_argument("--input", required=True, type=str, help="File containing the query-document pairs to judge with LLM")
    parser.add_argument("--prompt_template", required=True, type=str, help="prompt template")
    parser.add_argument("--prompts_file", required=False, default="", type=str, help="file to save prompts for all pairs")
    parser.add_argument("--result", required=True, type=str, help="File to save the LLM output")
    parser.add_argument("--output_mode", required=False, default="object", type=str, help="LLM output mode can be either: JSON or object")
    parser.add_argument("--prompt_type", required=False, default="user", type=str, help="The type of the template: can be system or user")
    parser.add_argument("--log_file", required=True, type=str, help="File to save the log")
    parser.add_argument("--max_tokens", required=False, type=int, default=150, help="max_tokens")
    parser.add_argument("--missing_file", required=False, type=str, default="", help="file to missing pairs to re-evaluate")
    parser.add_argument("--few_shots_file", required=False, type=str, default="", help="file contains few shot examples")
    parser.add_argument("--sleep", required=False, action="store_true", default=False, help="flag to activate 10 seconds delay between the requests")
    parser.add_argument("--overwrite", required=False, action="store_true", default=False, help="flag to override the content of the results file and make new content")
    parser.add_argument("--judged_pool", required=False, type=str, default="", help="Path to judged pool to copy pairs judged from if any")
    args = parser.parse_args()


    # model = "gpt-4o-2024-11-20"
    model = args.model
    input = args.input
    prompt_template = args.prompt_template
    prompts_file = args.prompts_file
    result = args.result
    log_file = args.log_file
    output_mode = args.output_mode
    prompt_type = args.prompt_type
    max_tokens = args.max_tokens
    missing_file = args.missing_file
    few_shots_file = args.few_shots_file
    sleep = args.sleep
    overwrite = args.overwrite
    judged_pool = args.judged_pool
    logger = utils.get_logger(log_file=log_file)


    if sleep:
        logger.info(f"sleep every 10 seconds is activated")

    gpt_judge = GPTJudge(model_name=model)
    prompt_template = toml.load(prompt_template)
    df_pairs  = utils.read_jsonl(input)

    missing_pairs = None
    sample_pairs = None
    if missing_file != "":
        missing_pairs = read_missing(missing_file)

    if overwrite:
        utils.write_line(prompts_file, "", mode='w')
        utils.write_line(result, "", mode='w')
    
    judged_pairs_dict = {}
    if judged_pool != "":
        judged_pairs_dict = copy_judged_pairs_from_pool(pool_path=judged_pool, df_pairs=df_pairs, result=result, logger=logger)


    if os.path.exists(result): # some pairs already judged and we need to exclude them
        df_res = pd.read_json(result, lines=True)
        for row in df_res.itertuples(index=False):
            judged_pairs_dict[(str(row.qid), str(row.docno))] = row._asdict()
        print(f"length of already judged pairs {len(judged_pairs_dict)}")
    
    cnt = 0
    missing = {}
    for row in df_pairs.itertuples():

        # if cnt > 1:
        #     break
        try:   
            qid = str(row.qid)
            docno = str(row.docno)
            q_text = row.q_text if "q_text" in df_pairs.columns else ""
            query_description = row.q_description
            segment = row.doc_text
            
            if (qid, docno) in judged_pairs_dict:
                print(f"Pair (qid={qid}, docno={docno}) is already judged")
                continue

            if sampling:
                in_sample = sample_pairs.get((qid, docno), 0)
                if in_sample == 0:
                    continue
            
            if missing_pairs is not None:
                in_missing= missing_pairs.get((qid, docno), 0)
                if in_missing == 0:
                    continue
            
            cnt +=1 
            if prompt_type == "user":
                user_prompt = prompt_template["user"].format(query=q_text, query_description=query_description, passage=segment)
                system_prompt = gpt_judge.get_system_prompt()
                # add few shots to the prompt
                if few_shots_file != "":
                    examples = utils.read_json(few_shots_file)
                    user_prompt = user_prompt.format(examples=examples)
            
            elif prompt_type == "system":
                system_prompt = prompt_template["system"]
                user_prompt = f'''# Query: << {q_text} >> \n  # Query description: << {query_description} >> \n  # Segment: << {segment} >>'''
                if few_shots_file != "":
                    examples = utils.read_json(few_shots_file)
                    system_prompt = system_prompt.format(examples=examples)

            elif prompt_type == "user_system":
                user_prompt = prompt_template['user'].format(query_description=query_description, segment=segment)
                system_prompt = prompt_template["system"]

            else:
                raise Exception 

            num_tokens = gpt_judge.num_tokens_from_string(system_prompt, model=model) + gpt_judge.num_tokens_from_string(user_prompt, model=model)
            logger.info(f"For the pair of qid = {qid}, docno ={docno}, the number of input tokens (system & user prompts) is {num_tokens}")

            # run the LLM 
            start_time = time.time()
            output = gpt_judge.run_gpt(user_prompt=user_prompt, system_prompt=system_prompt, 
                                       output_mode=output_mode, max_new_tokens=max_tokens, logger=logger)
            
            exec_time = time.time() - start_time

            if output == "":
                missing.update({(str(qid), str(docno)): 1})
                continue


            logger.info(f"Time for judging the pair of qid = {qid}, docno ={docno} is {exec_time:.3f} seconds and number of input tokens (system & user prompts) is {num_tokens}")

            if prompts_file != "":
                utils.write_line(prompts_file, json.dumps(user_prompt) + '\n', mode='a')

            if output_mode == "JSON":
                line = json.dumps({"qid": qid, "docno": docno, "relevance": output['relevance'], 
                                    'reason': output['reason'], "exec_time": exec_time}) + '\n'
            elif output_mode == "object":
                line = json.dumps({"qid": qid, "docno": docno, "relevance": output.relevance, 
                                    'reason': output.reason, "exec_time": exec_time}) + '\n'

            utils.write_line(result, line, mode='a')

            if sleep: # 10 seconds delay between the requests
                time.sleep(10) 

        except Exception as e:
            logger.error(f"When processing query id qid = {qid}, docno ={docno}, we got the following error: {e}")
            continue
        
    logger.info(f"missing = {missing}")


    
    

if __name__ == "__main__":
    main()



