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
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

origin_vote_prompt = '''Given an problem and several choices of partial solution, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
'''

# origin_vote_prompt = '''Given an problem and several choices of partial solution, decide which choice is most promising. Directly give the choice in the last line "The best choice is {s}", where s the integer id of the choice.
# '''


# Convert int64 to int before serializing to JSON
def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    raise TypeError

def vote_prompt_wrap(x: str, ys: list) -> str:
    prompt = origin_vote_prompt
    prompt += f'Problem:\n{x}\n'
    for i, y in enumerate(ys, 1):
        prompt += f'Choice {i}:\n{y}\n'
    return prompt

def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
    vote_results = [0 for _ in range(n_candidates)]
    for vote_output in vote_outputs:
        # print(vote_output)
        pattern = r".*best choice is .*(\d+).*"
        match = re.match(pattern, vote_output, re.DOTALL)
        if match:
            vote = int(match.groups()[0]) - 1
            if vote in range(n_candidates):
                vote_results[vote] += 1
        else:
            vote = random.choice(range(n_candidates))
            vote_results[vote] += 1
            print("random selection")
    return vote_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/reclor_val.json')
    parser.add_argument('--output_dir', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=2)
    parser.add_argument('--num_rollout', type=int, default=8)
    parser.add_argument('--num_foresight', type=int, default=4)
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--strategy', type=str, default='tot_vote')
    parser.add_argument('--file_name', type=str, default='241225-21')
    parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    args = parser.parse_args()

    # args.output_path = args.output_dir + args.strategy  + '.json'
    args.output_path = args.output_dir + args.file_name  + '.json'

    if args.model_id=="llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    elif args.model_id=="qwen2.5":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
    elif args.model_id=="mistral":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de"
    elif args.model_id=="gemma":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
    elif args.model_id=="llama70B":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/6100a3f3f907fd2bb80ed0e79c1f170ade0bc6ce"

    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, max_length=2048, trust_remote_code=True)


    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token
    start_time = time.time()
    total_rollout_times = 0
    saved_rollout_times = 0
    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True)
    # model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=1, trust_remote_code=True, max_model_len=4096)

    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size
    
    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)
    # test_data = test_data[295:]
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
         try:
            # 对于每一个问题
            # if iadx == 1:
            #     break
            problem_start_time = time.time()
            output_token_num_for_this_question = 0
            input_token_num_for_this_question = 0
            iadx += 1
            traj_pool = [[] for _ in range(num_foresight)]
            step_pool = [[] for _ in range(num_foresight)]
            prob_pool = [[] for _ in range(num_foresight+1)]
            adv_pool = [[] for _ in range(num_foresight+1)]
            sample_id_pool = []
            
            traj_complete = False
            previous_steps_list = ["The reasoning steps are:\n\n" for _ in range(step_beam_size)]
            previous_q_value_list = [0.0 for _ in range(step_beam_size)]
            T = 0
            for T in range(num_foresight):
                # 对每个问题foresight次数，达到foresight次数后，剩下的直接补全
                reasoning_steps_list = previous_steps_list

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

                # fake_input = tokenizer.apply_chat_template(
                #     chat,
                # )
                # input_token_num_for_this_question += len(fake_input)
                inputs = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False, # 输出的是 str，而不是token_ids
                )
                # print(inputs)
                inputs = inputs.replace(stop_token, "").strip()
                
                inputs_list = [inputs + reasoning_steps_list[beam_idx] for beam_idx in range(step_beam_size)] # 对每个step beam size进行

                for each_input in inputs_list:
                    input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])


                sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["\n", "<end_of_reasoning>"])
                # sampling_params = SamplingParams(max_tokens=1024 ,n=num_rollout, logprobs=0, best_of=4, temperature=0, use_beam_search=True, stop=["\n", "<end_of_reasoning>"])
                # 根据当前的状态，foresight
                outputs = model.generate(inputs_list, sampling_params)
                total_rollout_times += step_beam_size * num_rollout
                

                selected_steps = []
                inputs_list = []
                candidates_list = []
                reasoning_steps_candidate_list = []
                for beam_idx in range(step_beam_size):
                    for j in range(num_rollout):
                        output = outputs[beam_idx].outputs[j]
                        response = output.text.strip()
                        selected_steps.append(response)
                        output_token_num_for_this_question += len(output.token_ids)
                        reasoning_steps_candidate = reasoning_steps_list[beam_idx] + "\n" + response
                        reasoning_steps_candidate_list.append(reasoning_steps_candidate)
                        candidates_list.append(response)
                

                if "gsm" in args.data_path or "math" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                # TODO: 下面的还没改 (已改)
                elif "reclor" in args.data_path or "logiqa" in args.data_path:
                    question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "strategy" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "cs" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "gpqa" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "arc" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "scibench" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "truthfulqa_mc1" in args.data_path:
                    question = test_data[i]['input']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "humaneval" in args.data_path:
                    question = test_data[i]['prompt']
                    vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': vote_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                if args.model_id=="mistral" or args.model_id=="gemma":
                    chat[1]['content']= system_prompt +"\n" + chat[1]['content']
                    chat =chat[1:]
                # chat = system_prompt + '\nThe question: ' + question + '\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n' + reasoning_steps_candidate
                inputs_list.append(tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    # add_generation_prompt=True
                ).rstrip(stop_token).rstrip())
                fake_input = tokenizer.apply_chat_template(
                    chat,
                )
                input_token_num_for_this_question += len(fake_input)
                    
                # TODO:确定一下vote次数
                n_vote = step_beam_size
                sampling_params = SamplingParams(max_tokens=4096 ,n=step_beam_size, logprobs=1)

                outputs = model.generate(inputs_list, sampling_params)
                total_rollout_times += len(inputs_list)

                normalized_logp_list = []
                # advantages_list = []
                vote_outputs = []
                # TODO: here
                for jaa in range(n_vote):
                    output = outputs[0].outputs[jaa]
                    response = output.text.strip()
                    vote_outputs.append(response)
                    output_token_num_for_this_question += len(output.token_ids)
                vote_res = vote_outputs_unwrap(vote_outputs, step_beam_size*step_beam_size)

                # 通过不同策略选择index
                if args.strategy == 'tot_vote':
                    arr = np.array(vote_res)
                    selected_index_list = arr.argsort()[-step_beam_size:][::-1].tolist()  # 先排序后取倒数stem beam size个，再反转顺序
                
                sample_id_pool.append(selected_index_list) # 相当于默认不使用max和random

                previous_steps_list_updated, previous_q_value_list = [], []
                for m, selected_index in enumerate(selected_index_list):
                    previous_steps_list_updated.append(previous_steps_list[selected_index//num_rollout] + candidates_list[selected_index].strip() + "\n")
                previous_steps_list = previous_steps_list_updated

            
            if "gsm" in args.data_path or "math" in args.data_path:
                question = test_data[i]['input']
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly output the reasoning steps.\n'},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}\n"},
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
                tokenize=False,
                add_generation_prompt=True
            ).rstrip(stop_token).rstrip()

            inputs_list = [inputs + previous_steps_list[beam_idx] for beam_idx in range(step_beam_size)]
            sampling_params = SamplingParams(max_tokens=3000 ,n=1, logprobs=0, stop="<end_of_reasoning>")
            for each_input in inputs_list:
                input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
            outputs = model.generate(inputs_list, sampling_params)
            total_rollout_times += step_beam_size

            normalized_logp_list = []
            # advantages_list = []
            candidates_list = []
            reasoning_steps_candidate_list = []
            vote_outputs = []
            for je in range(step_beam_size):
                output = outputs[je].outputs[0]
                response = output.text.strip()
                candidates_list.append(response)
                normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                # advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[j//num_rollout])
                prob_pool[T+1].append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                # adv_pool[T+1].append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[j//num_rollout])
                reasoning_steps_candidate = reasoning_steps_list[je] + "\n" + response
                reasoning_steps_candidate_list.append(reasoning_steps_candidate)
                output_token_num_for_this_question += len(output.token_ids)
                vote_outputs.append(response)


            if "gsm" in args.data_path or "math" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            # TODO: 下面的还没改
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "strategy" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "cs" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "gpqa" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "arc" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "scibench" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "truthfulqa_mc1" in args.data_path:
                question = test_data[i]['input']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "humaneval" in args.data_path:
                question = test_data[i]['prompt']
                vote_prompt = vote_prompt_wrap(question, reasoning_steps_candidate_list)
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': vote_prompt},
                    {'role': 'assistant', 'content': ''}
                ]
            if args.model_id=="mistral" or args.model_id=="gemma":
                chat[1]['content']= system_prompt +"\n" + chat[1]['content']
                chat =chat[1:]
            # chat = system_prompt + '\nThe question: ' + question + '\nPlease directly output the reasoning steps.\nThe reasoning steps are:\n' + reasoning_steps_candidate
            inputs_list.append(tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                # add_generation_prompt=True
            ).rstrip(stop_token).rstrip())
            fake_input = tokenizer.apply_chat_template(
                chat,
            )
            input_token_num_for_this_question += len(fake_input)
            # TODO:确定一下vote次数
            n_vote2 = 1
            sampling_params = SamplingParams(max_tokens=1024 ,n=n_vote2, logprobs=1)

            outputs = model.generate(inputs_list, sampling_params)
            total_rollout_times += len(inputs_list)

            normalized_logp_list = []
            # advantages_list = []
            vote_outputs = []
            # TODO: here
            for j in range(n_vote2):
                output = outputs[0].outputs[j]
                response = output.text.strip()
                vote_outputs.append(response)
                output_token_num_for_this_question += len(output.token_ids)
            vote_res = vote_outputs_unwrap(vote_outputs, step_beam_size)

            if args.strategy == 'tot_vote':
                arr = np.array(vote_res)
                selected_index_final = arr.argsort()[-1:][::-1][0]
            all_output_token_num += output_token_num_for_this_question
            all_input_token_num += input_token_num_for_this_question
            print(f"question {i} output token num: {output_token_num_for_this_question}")
            print(f"question {i} input token num: {input_token_num_for_this_question}")
            print(f"all output token num: {all_output_token_num}")
            print(f"all input token num: {all_input_token_num}")
            sample_id_pool.append([selected_index_final for _ in range(step_beam_size)])
            whole_traj = previous_steps_list[selected_index_final] + "\n" + candidates_list[selected_index_final] # 最终答案的traj
            whole_traj_list = [previous_steps_list[beam_idx] + "\n" + candidates_list[beam_idx] for beam_idx in range(step_beam_size)] # 每个beam的traj

            ###########################################
            #           Write to result file          #
            ###########################################
            result = {}
            result['id'] = i
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
                
            result['response'] = whole_traj
            result['response_all_beams'] = whole_traj_list
            probelm_stop_time = time.time()
            print(f"problem {i} time usage: {probelm_stop_time - problem_start_time}")

            if args.record_process:
                result['foresight_steps'] = T + 1
                result['traj_pool'] = traj_pool
                result['step_pool'] = step_pool
                result['prob_pool'] = prob_pool
                result['adv_pool'] = adv_pool
                result['sample_id_pool'] = sample_id_pool

            f.write(json.dumps(result, default=convert) + '\n')
            f.flush() 
            # except:
            #     f.write(json.dumps({'id': i, 'response': 'error'}) + '\n')
            #     f.flush()
         except:
            with open('o1/cluster_results/over_length_results/'+args.file_name + '.txt', 'a') as fd:
                fd.write(json.dumps({'id': i, 'response': 'error'}) + '\n')
                fd.flush()
    end_time = time.time()
    time_span = end_time - start_time
    print(f"time: {time_span}")
    time_path = args.time_path + args.file_name + '.txt'
    with open(time_path, 'a') as f:
        f.write('time:  ' + str(time_span) + '\n')
        f.write('total:  ' + str(total_rollout_times) + '\n')
        f.write('save:  ' + str(saved_rollout_times) + '\n')
        f.write('num_rollout:  ' + str(num_rollout) + '\n')
        f.write('num_foresight:  ' + str(num_foresight) + '\n')
        f.write('all_output_token_num:  ' + str(all_output_token_num) + '\n')
        f.write('all_input_token_num:  ' + str(all_input_token_num) + '\n')
    print('total: ', total_rollout_times)
    print('save: ', saved_rollout_times)