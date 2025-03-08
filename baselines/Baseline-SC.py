import json
import random
import numpy as np
import torch
import os
import argparse
import time
from data.math_example import MATH_POT_FEW_SHOT, MATH_COT_FEW_SHOT, GSM_COT_8_SHOT, MATH_COT_4_SHOT
from data.logic_example import LOGIC_MRC_COT_4_SHOT
from vllm import LLM, SamplingParams
from transformers import (
    StoppingCriteriaList,
    StoppingCriteria,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
import re
INF = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/reclor_val.json')
    parser.add_argument('--output_dir', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=4)  # sc baseline用 num rollout来进行控制
    parser.add_argument('--num_foresight', type=int, default=4)
    parser.add_argument('--record_process', type=bool, default=False)
    parser.add_argument('--strategy', type=str, default='test')
    parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    parser.add_argument('--file_name', type=str, default='test')
    args = parser.parse_args()
    ffname = args.file_name
    args.output_path = args.output_dir + ffname  + '.json'
    ffname = args.file_name
    if args.model_id=="llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    elif args.model_id=="qwen2.5":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
    elif args.model_id=="mistral":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de"
    elif args.model_id=="gemma":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"

    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, max_length=2048, trust_remote_code=True)


    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    start_time = time.time()
    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
    # model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True, max_model_len=4096)

    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size
    
    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)

    OUTPUT_PATH = args.output_path
    if "gsm" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}"
    elif "math" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
    elif "reclor" in args.data_path or "logiqa" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Please finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{LOGIC_MRC_COT_4_SHOT}\n"
    elif "strategy" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with one question. At the end, you must output 'Yes' or 'No' after 'The answer is: '."
    elif "cs" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are five options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response."
    elif "gpqa" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response."
    elif "arc" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response."
    elif "scibench" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>."
    elif "truthfulqa_mc1" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a question. There are multiple options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response."
    elif "humaneval" in args.data_path:
        system_prompt = f"Please directly complete the following code without any additional comments.\n"    
    
    iadx = 0
    all_output_token_num = 0
    all_input_token_num = 0

    with open(OUTPUT_PATH, "w") as f:
        for i in range(len(test_data)):
            # if i > 3:
            #     break
            try_time = 0
            while try_time < 3:
                try:
                    # 对于每一个问题
                    # if iadx == 1:
                    #     break
                    problem_start_time = time.time()
                    output_token_num_for_this_question = 0
                    input_token_num_for_this_question = 0
                    iadx += 1
                    result = {}
                    result['id'] = i
                    result['response'] = ''
                    all_res = []

                    if "gsm" in args.data_path or "math" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "reclor" in args.data_path or "logiqa" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "strategy" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'The question is: ' + test_data[i]['input'] + "\nAt the end, you must output 'Yes' or 'No' after 'The answer is: '." + '\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "cs" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' + test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "gpqa" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' + test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "arc" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' + test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "scibench" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'The question: ' + test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "truthfulqa_mc1" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'The question: ' + test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "humaneval" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': test_data[i]['prompt'] + "\n"},
                            {'role': 'assistant', 'content': ''}
                        ]
                    if args.model_id=="mistral" or args.model_id=="gemma":
                        chat[1]['content']= system_prompt +"\n" + chat[1]['content']
                        chat =chat[1:]

                    inputs = tokenizer.apply_chat_template(
                        chat,
                        tokenize=False, # 输出的是 str，而不是token_ids
                    )
                    # print(inputs)
                    inputs = inputs.replace(stop_token, "").strip()
                    
                    inputs_list = [inputs] # 对每个step beam size进行

                    for each_input in inputs_list:
                        input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])


                    sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["<end_of_reasoning>"])

                    outputs = model.generate(inputs_list, sampling_params)

                    for _ in range(num_rollout):
                        output = outputs[0].outputs[_]
                        response = output.text.strip()
                        output_token_num_for_this_question += len(output.token_ids)
                        all_res.append(response)
                    break
                except:
                    try_time += 1
                    if try_time == 3:
                        with open('/cpfs01/user/xufangzhi/o1/cluster_results/over_length_results/'+ffname+'.txt', 'a') as fsa:
                            fsa.write(str(i))
                            fsa.write('\n')
            all_input_token_num += input_token_num_for_this_question
            all_output_token_num += output_token_num_for_this_question
            print(f"question {i} input token num: {input_token_num_for_this_question}")
            print(f"question {i} output token num: {output_token_num_for_this_question}")
            print(f"all input token num: {all_input_token_num}")
            print(f"all output token num: {all_output_token_num}")


            if "gsm" in args.data_path or "math" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                result['question'] = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                result['ground_truth'] = test_data[i]['label'] if "label" in test_data[i] else None
            elif "strategy" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "cs" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "gpqa" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "arc" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "scibench" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "truthfulqa_mc1" in args.data_path:
                result['question'] = test_data[i]['input']
                result['ground_truth'] = test_data[i]['target']
            elif "humaneval" in args.data_path:
                result['question'] = test_data[i]['prompt']
                result['task_id'] = test_data[i]['task_id']
                result['completion'] = response

            result['response_all_beams'] = all_res
            f.write(json.dumps(result) + '\n')
            f.flush() 
        end_time = time.time()
        time_span = end_time - start_time
        print(f"time: {time_span}")
        time_path = args.time_path + ffname + '.txt'
        with open(time_path, 'a') as f:
            f.write('time:  ' + str(time_span) + '\n')
            f.write('num_rollout:  ' + str(num_rollout) + '\n')
            f.write('num_foresight:  ' + str(num_foresight) + '\n')
            f.write('all_output_token_num:  ' + str(all_output_token_num) + '\n')
            f.write('all_input_token_num:  ' + str(all_input_token_num) + '\n')
