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
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# origin_vote_prompt = '''Given an problem and several choices of partial solution, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
# '''

evaluate_prompt_dict = {'gsm':'''
Q: Julie climbed 15 steps up the giant slide. She climbed down 6 steps to talk to her friend, Maria. Then she climbed up 8 steps to get to the top. How many steps does the slide have?

A:
Julie climbed 15 steps up.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
Then she climbed down 6 steps.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
Then she climbed up 8 steps.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So she climbed 15 + 8 = 23 steps.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (B), because she also climbed down 6 steps, so she climbed 23 - 6 = 17 steps.
So the slide has 23 steps.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of steps of slides is incorrect.
So the answer is 23.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of steps of slides is incorrect.





Q: Suzanne read the first 15 pages of her book on Monday. She read 16 more pages than that on Tuesday. Then there were 18 pages left. How many pages are in Suzanne's book altogether?

A:
Suzanne read 15 pages on Monday.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
Then she read 16 more pages on Tuesday.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So she read 15 + 16 = 31 pages in total.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (B), because she read 16 more pages than that on Tuesday, so she read 15 + 16 = 31 pages on tuesday. So she read 15 + 31 = 46 pages in total.
Then there were 18 pages left.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of total read pages of monday and tuesday is incorrect.
So the book had 31 + 18 = 49 pages.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of total read pages of monday and tuesday is incorrect. So the book had 46 + 18 = 64 pages.
So the answer is 49.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of total read pages of monday and tuesday is incorrect.





Q: Allison brought some CDs online. Each CD cost $7. There was an additional charge of $4 per order for shipping costs. The total bill came to $60. How many CDs did Allison buy? 

A:
Each CD cost 7 dollars.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
And there was an additional charge of 4 dollars.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So the total cost of each CD is 7 + 4 = 11 dollars.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (B), because each CD cose 7 dollars.
So 60 / 11 = 5.45.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (B), because it cost 4 dollars for shipping costs. So the cost of CDs is 60 - 4 = 56 dollars. So Allison bought 56 / 7 = 8 CDs.
So the answer is 5.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of number of CDs is incorrect.





Q: Luis and Cameron shared some stickers is the ratio 5:2. Luis received 15 more stickers than Cameron. How many stickers were there altogether?

A:
Let's say there were x stickers.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
Then Luis got 5x/7 and Cameron got 2x/7.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
Luis got 15 more than Cameron, so 5x/7 - 2x/7 = 15.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So 3x/7 = 15.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So x = 105.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (B), because 3x/7 = 15. So x = 15 * 7 / 3 = 35. So there were 35 stickers.
So the answer is 105.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A), but the value of number of stickers is incorrect.





Q: Alexa has 92 cents in her pocket. She wants to buy 3 pencils at the school supply store. Each pencil costs 8 cents. How much money will Alexa have left?

A:
Alexa has 92 cents.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
And 3 pencils for 8 cents each will be 3 * 8 = 24 cents.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So she has 92 - 24 = 68 cents left.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
So the answer is 68.
# Is the above step of reasoning:
# (A) Correct
# (B) Incorrect
# The above step of reasoning is (A)
'''
}

def vote_prompt_wrap(x: str, ys: list) -> str:
    prompt = origin_vote_prompt
    prompt += f'Problem:\n{x}\n'
    for i, y in enumerate(ys, 1):
        prompt += f'Choice {i}:\n{y}\n'
    return prompt

def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
    vote_results = [0 for _ in range(n_candidates)]
    for vote_output in vote_outputs:
        pattern = r".*best choice is .*(\d+).*"
        match = re.match(pattern, vote_output, re.DOTALL)
        if match:
            vote = int(match.groups()[0]) - 1
            if vote in range(n_candidates):
                vote_results[vote] += 1
        else:
            # print(f'vote no match: {[vote_output]}')
            print(f'vote no match')
    return vote_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/math_500_test.json')
    parser.add_argument('--output_dir', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=4)
    parser.add_argument('--num_foresight', type=int, default=4)
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--file_name', type=str, default='guided_decoding')
    parser.add_argument('--strategy', type=str, default='guided_decoding')
    parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
    args = parser.parse_args()

    args.output_path = args.output_dir + args.file_name  + '.json'

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
    # evaluate_prompt = evaluate_prompt_dict[args.datasets]
    evaluate_prompt = evaluate_prompt_dict["gsm"]
    with open(OUTPUT_PATH, "w") as f:
        for i in range(len(test_data)):
        #   try:
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
            
            # 给guided decoding用的
            choice_prefix = ['# Is the above step of reasoning:',
                '# (A) Correct', '# (B) Incorrect', '# The above step of reasoning is:']
            steps_text = [[] for _ in range(step_beam_size)]
            steps_cumulative_logp = [[] for _ in range(step_beam_size)]
            steps_len = [[] for _ in range(step_beam_size)]
            steps_eval_text = [[] for _ in range(step_beam_size)] # 这里面没有The reasoning steps are:\n\n这句话
            # steps_eval_logp = [[] for _ in range(step_beam_size)]
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
                outputs = model.generate(inputs_list, sampling_params)
                total_rollout_times += step_beam_size * num_rollout
                

                selected_steps = []
                inputs_list = []
                candidates_list = []
                reasoning_steps_candidate_list = []
                # 用于 gd
                tem_steps_text = []
                tem_steps_cumulative_logp = []
                tem_steps_len = []
                for beam_idx in range(step_beam_size):
                    for j in range(num_rollout):
                        output = outputs[beam_idx].outputs[j]
                        response = output.text.strip()
                        selected_steps.append(response)
                        output_token_num_for_this_question += len(output.token_ids)
                        reasoning_steps_candidate = reasoning_steps_list[beam_idx] + "\n" + response
                        reasoning_steps_candidate_list.append(reasoning_steps_candidate)
                        candidates_list.append(response)
                        tem_steps_text.append(response)
                        tem_steps_cumulative_logp.append(output.cumulative_logprob)
                        tem_steps_len.append(len(output.token_ids))

                if "reclor" in args.data_path or "logiqa" in args.data_path:
                    question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                elif "humaneval" in args.data_path:
                    question = test_data[i]['prompt']
                else:
                    question = test_data[i]['input']
                # 构建eval的prompt
                evaluate_prompt_list = []
                
                for ijdx in range(len(candidates_list)):
                    tem_prompt = ''
                    # tem_prompt = '' + evaluate_prompt # TODO: evaluate_prompt 需要根据不同任务进行修改
                    # tem_prompt += '\n\n\n\n\n'
                    # tem_prompt += f'Solve and evaluate this problem following previous examples:\nQ:{question}\n\nA:'
                    tem_prompt += f'Evaluate this problem following previous examples:\nQ:{question}\n\nA:\n'
                    history_steps_text_list = steps_text[ijdx//num_rollout]
                    history_steps_eval_text_list = steps_eval_text[ijdx//num_rollout]
                    for text, eval_text in zip(history_steps_text_list, history_steps_eval_text_list):
                        tem_prompt += text
                        if tem_prompt[-1] != '\n':
                            tem_prompt += '\n'
                        for prefix in choice_prefix:
                            tem_prompt += prefix + '\n'
                        # 去掉最后一个\n
                        tem_prompt = tem_prompt[:-1]
                        tem_prompt += ' ' + eval_text + '\n'
                    tem_prompt += tem_steps_text[ijdx] + '\n'
                    for prefix in choice_prefix:
                        tem_prompt += prefix + ' \n'
                    tem_prompt = tem_prompt[:-1]
                    evaluate_prompt_list.append(tem_prompt)

                inputs_list = []
                for each_eval_prompt in evaluate_prompt_list:
                    if "gsm" in args.data_path or "math" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    # TODO: 下面的还没改 (已改)
                    elif "reclor" in args.data_path or "logiqa" in args.data_path:
                        question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]

                    elif "strategy" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "cs" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "gpqa" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "arc" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "scibench" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "truthfulqa_mc1" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "humaneval" in args.data_path:
                        question = test_data[i]['prompt']
                        chat = [
                            {'role': 'system', 'content': evaluate_prompt}, # 这里因为是评估问题，就不给求解问题的system prompt了
                            {'role': 'user', 'content': each_eval_prompt},
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
                    
                # 每个candidate只评估一次
                n_vote = 1
                sampling_params = SamplingParams(max_tokens=4096 ,n=n_vote, logprobs=1, stop=[' Correct', ' Incorrect',  "<end_of_reasoning>"])

                outputs = model.generate(inputs_list, sampling_params)
                total_rollout_times += len(inputs_list)

                normalized_logp_list = []
                tem_eval_text_list = []
                tem_judge_token_p_list = []

                # TODO: here
                for jaa in range(len(inputs_list)):
                    output = outputs[jaa].outputs[0]
                    response = output.text.strip()
                    processed_response = response
                    # processed_response = response.split('\n')[-1]
                    # processed_response = processed_response.split('The above step of reasoning is:')[-1]
                    if 'A' in processed_response:
                        processed_response = '(A)'
                    else:
                        processed_response = '(B)'
                    tem_eval_text_list.append(processed_response)
                    tem_p = 0
                    # 反向遍历output.logprobs 找到那个A或者B
                    for token_logp in output.logprobs[::-1]:
                        find = False
                        for k, v in token_logp.items():
                            if 'A' in v.decoded_token or 'B' in v.decoded_token: # 因为有的时候可能是 A) 这种情况
                                tem_logp = v.logprob
                                tem_p = np.exp(tem_logp)
                                find = True
                                break
                        if find:
                            break
                    if 'B' in processed_response:
                        tem_p = 1 - tem_p
                    tem_judge_token_p_list.append(tem_p)
                    output_token_num_for_this_question += len(output.token_ids)
                
                # 评估
                # 历史平均的 exp(logp)**0.5 * exp(judge_token_logp)**0.5
                weights = []
                for jidx in range(len(tem_steps_cumulative_logp)):
                    cumulative_logp, len_token, judge_p = 0, 0, 0
                    for kk in range(T): # 如果T是0，那么这个循环不会执行
                        cumulative_logp += steps_cumulative_logp[jidx//num_rollout][kk]
                        len_token += steps_len[jidx//num_rollout][kk]
                    cumulative_logp += tem_steps_cumulative_logp[jidx]
                    len_token += tem_steps_len[jidx]
                    cumulative_p = np.exp(cumulative_logp / (len_token+1e-8)) ** 0.5
                    judge_p = tem_judge_token_p_list[jidx] ** 0.5
                    weights.append(cumulative_p * judge_p)
                weights = np.array(weights)
                # weights = softmax(weights)
                def softmax(x):
                    return np.exp(x) / np.sum(np.exp(x))
                weights = softmax(weights)
                # 根据weights随机挑选step beam size 个
                selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
    
                sample_id_pool.append(selected_index_list) # 相当于默认不使用max和random

                previous_steps_list_updated, previous_q_value_list = [], []
                for m, selected_index in enumerate(selected_index_list):
                    previous_steps_list_updated.append(previous_steps_list[selected_index//num_rollout] + candidates_list[selected_index].strip() + "\n")
                previous_steps_list = previous_steps_list_updated

                # 更新 steps_text, steps_cumulative_logp, steps_len, steps_eval_text
                copy_steps_text = steps_text.copy()
                copy_steps_cumulative_logp = steps_cumulative_logp.copy()
                copy_steps_len = steps_len.copy()
                copy_steps_eval_text = steps_eval_text.copy()
                for m, selected_index in enumerate(selected_index_list):
                    # steps_text[m] = steps_text[selected_index//num_rollout] + [tem_steps_text[selected_index]]
                    # steps_cumulative_logp[m] = steps_cumulative_logp[selected_index//num_rollout] + [tem_steps_cumulative_logp[selected_index]]
                    # steps_len[m] = steps_len[selected_index//num_rollout] + [tem_steps_len[selected_index]]
                    # steps_eval_text[m] = steps_eval_text[selected_index//num_rollout] + [tem_eval_text_list[selected_index]]
                    steps_text[m] = copy_steps_text[selected_index//num_rollout] + [tem_steps_text[selected_index]]
                    steps_cumulative_logp[m] = copy_steps_cumulative_logp[selected_index//num_rollout] + [tem_steps_cumulative_logp[selected_index]]
                    steps_len[m] = copy_steps_len[selected_index//num_rollout] + [tem_steps_len[selected_index]]
                    steps_eval_text[m] = copy_steps_eval_text[selected_index//num_rollout] + [tem_eval_text_list[selected_index]]
            
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
            # 用于 gd
            tem_steps_text = []
            tem_steps_cumulative_logp = []
            tem_steps_len = []
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
                tem_steps_text.append(response)
                tem_steps_cumulative_logp.append(output.cumulative_logprob)
                tem_steps_len.append(len(output.token_ids))

            # 评估
            if "reclor" in args.data_path or "logiqa" in args.data_path:
                question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
            elif "humaneval" in args.data_path:
                question = test_data[i]['prompt']
            else:
                question = test_data[i]['input']
            # 构建eval的prompt
            evaluate_prompt_list = []
            for ijdx in range(len(candidates_list)):
                tem_prompt = ''
                tem_prompt += f'Evaluate this problem following previous examples:\nQ:{question}\n\nA:\n'
                history_steps_text_list = steps_text[ijdx]
                history_steps_eval_text_list = steps_eval_text[ijdx]
                for text, eval_text in zip(history_steps_text_list, history_steps_eval_text_list):
                    tem_prompt += text
                    if tem_prompt[-1] != '\n':
                        tem_prompt += '\n'
                    for prefix in choice_prefix:
                        tem_prompt += prefix + '\n'
                    # 去掉最后一个\n
                    tem_prompt = tem_prompt[:-1]
                    tem_prompt += ' ' + eval_text + '\n'
                tem_prompt += tem_steps_text[ijdx] + '\n'
                for prefix in choice_prefix:
                    tem_prompt += prefix + ' \n'
                tem_prompt = tem_prompt[:-1]
                evaluate_prompt_list.append(tem_prompt)

            inputs_list = []
            for each_eval_prompt in evaluate_prompt_list:
                if "gsm" in args.data_path or "math" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                # TODO: 下面的还没改
                elif "reclor" in args.data_path or "logiqa" in args.data_path:
                    question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "strategy" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "cs" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "gpqa" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "arc" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "scibench" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "truthfulqa_mc1" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "humaneval" in args.data_path:
                    question = test_data[i]['prompt']
                    chat = [
                        {'role': 'system', 'content': evaluate_prompt},
                        {'role': 'user', 'content': each_eval_prompt},
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
            n_vote = 1
            sampling_params = SamplingParams(max_tokens=1024 ,n=n_vote, logprobs=1)

            outputs = model.generate(inputs_list, sampling_params)
            total_rollout_times += len(inputs_list)

            normalized_logp_list = []
            tem_eval_text_list = []
            tem_judge_token_p_list = []
            # TODO: here
            for jaa in range(step_beam_size):
                output = outputs[jaa].outputs[0]
                response = output.text.strip()
                processed_response = response
                # processed_response = response.split('\n')[-1]
                # processed_response = processed_response.split('The above step of reasoning is:')[-1]
                if 'A' in processed_response:
                    processed_response = '(A)'
                else:
                    processed_response = '(B)'
                tem_eval_text_list.append(processed_response)
                tem_p = 0
                # 反向遍历output.logprobs 找到那个A或者B
                for token_logp in output.logprobs[::-1]:
                    find = False
                    for k, v in token_logp.items():
                        if 'A' in v.decoded_token or 'B' in v.decoded_token: # 因为有的时候可能是 A) 这种情况
                            tem_logp = v.logprob
                            tem_p = np.exp(tem_logp)
                            find = True
                            break
                    if find:
                        break
                if 'B' in processed_response:
                    tem_p = 1 - tem_p
                tem_judge_token_p_list.append(tem_p)

                output_token_num_for_this_question += len(output.token_ids)
            
            # 评估
            # 历史平均的 exp(logp)**0.5 * exp(judge_token_logp)**0.5
            weights = []
            for jidx in range(step_beam_size):
                cumulative_logp, len_token, judge_p = 0, 0, 0
                for kk in range(num_foresight):
                    cumulative_logp += steps_cumulative_logp[jidx][kk]
                    len_token += steps_len[jidx][kk]
                cumulative_p = np.exp(cumulative_logp / (len_token+1e-8)) ** 0.5
                judge_p = np.exp(tem_judge_token_p_list[jidx]) ** 0.5
                weights.append(cumulative_p * judge_p)
            weights = np.array(weights)
            # weights = softmax(weights)
            def softmax(x):
                return np.exp(x) / np.sum(np.exp(x))
            weights = softmax(weights)
            # 根据weights随机挑选1个
            selected_index_final = np.random.choice(len(weights), p=weights)
            
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
                result['question'] = question
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
            f.write(json.dumps(result) + '\n')
            f.flush() 
        #   except:
        #     f.write(json.dumps({'id': i, 'response': 'error'}) + '\n')
        #     f.flush()
    print("average output token num: ", all_output_token_num / len(test_data))
    print("average input token num: ", all_input_token_num / len(test_data))
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