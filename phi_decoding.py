# This version is the orignal code, so it may be a little difficult to successfully run.
import os
import json
import random
import numpy as np
import torch
import argparse
from data.math_example import MATH_POT_FEW_SHOT, MATH_COT_FEW_SHOT, GSM_COT_8_SHOT, MATH_COT_4_SHOT
from data.logic_example import LOGIC_MRC_COT_4_SHOT
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

INF = 10
import time
temp=0.1
def softmax(x):
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, default='gsm')
    parser.add_argument('--model_id', type=str, default='llama3.1')
    parser.add_argument('--data_path', type=str, default='data/gsm_test.json')
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--step_beam_size', type=int, default=4)
    parser.add_argument('--num_rollout', type=int, default=4)
    parser.add_argument('--num_foresight', type=int, default=8)
    parser.add_argument('--record_process', type=bool, default=True)
    parser.add_argument('--strategy', type=str, default='cluster')
    parser.add_argument('--time_path', type=str, default='results/time/')
    parser.add_argument('--final_select_strategy', type=str, default='same_as_strategy')
    parser.add_argument('--pruning_strategy', type=str, default='sirno_low_sigma') # none, sirno_low_sigma, advno_low_sigma, advno_above_avg, adv
    parser.add_argument('--sigma_rate', type=float, default=1.0)
    parser.add_argument('--depth_pruning_strategy', type=str, default='cluster') # early stopping [around_mean, within_sigma]
    parser.add_argument('--threshold', type=float, default=0.69) # threshold for early stopping
    parser.add_argument('--least_foresight_num', type=int, default=4)
    parser.add_argument('--adv_above0', type=str, default='True')
    parser.add_argument('--file_name', type=str, default='orignal_code')
    parser.add_argument('--cluster_num', type=int, default=3)
    parser.add_argument('--gpus', type=int, default=2)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=1)
    args = parser.parse_args()
    # device = "cuda:3"
    start_time = time.time()
    np.random.seed(args.seed)
    ffname = args.file_name
    args.output_path = args.output_dir + ffname + '.json'

    if args.model_id=="llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = "/data/xrm/yh/test_unsloth/Meta-Llama-3.1-8B-Instruct"
        # PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
    elif args.model_id=="qwen7B":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75"
    elif args.model_id=="qwen3B":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    # elif args.model_id=="gemma2B":
    #     PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2b/snapshots/2ac59a5d7bf4e1425010f0d457dde7d146658953"
    elif args.model_id=="mistral":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de"
    elif args.model_id=="gemma":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2b/snapshots/2ac59a5d7bf4e1425010f0d457dde7d146658953"
        # PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2-9b-it/snapshots/11c9b309abf73637e4b6f9a3fa1e92e615547819"
    elif args.model_id=="llama70B":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/6100a3f3f907fd2bb80ed0e79c1f170ade0bc6ce"
    elif args.model_id=="r1-qwen-7b":
        PATH_TO_CONVERTED_WEIGHTS = '/nas/shared/NLP_A100/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/393119fcd6a873e5776c79b0db01c96911f5f0fc/'
    elif args.model_id=="r1-llama-8b":
        PATH_TO_CONVERTED_WEIGHTS = '/nas/shared/NLP_A100/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/81cee02dd020268dced5fa1327e8555acce9c63c/'
    elif args.model_id=="241219-2":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/241219-2_llama3.1_8B/"
    elif args.model_id=="241202-3":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/241202-3_llama3.1_8B/"
    elif args.model_id=="250107-2":
        PATH_TO_CONVERTED_WEIGHTS = "/nas/shared/NLP_A100/xufangzhi/symbol-llm-omni/open-instruct/output/250107-2_qwen2.5_8B/"
    
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, max_length=32768, trust_remote_code=True)
    
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    saved_rollout_times = 0
    total_rollout_times = 0
    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=args.gpus,  max_model_len=32768,trust_remote_code=True)

    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size
    
    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)

    # test_data = test_data[args.start:args.end]


    OUTPUT_PATH = args.output_path
    prompt_ = "A conversation between user and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."

    if "gsm" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}"
    elif "math" in args.data_path or "aime" in args.data_path :
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

    print('\n\n\n\n\nstrategy\n')
    print(args.strategy)
    iadx = 0
    import time
    all_output_token_num = 0
    all_input_token_num = 0
    with open(OUTPUT_PATH, "w") as f:
        print('len test data', len(test_data))
        all_traj_info = []

        for i in range(len(test_data)):
            try_time = 0
            # if i >1:
            #     break
            while try_time < 3:
                # try:
                    # if i <= 436:
                    #     continue
                    # if iadx == 1:
                    #     break
                    stop_foresight = False
                    iadx += 1
                    output_token_num_for_this_question = 0
                    input_token_num_for_this_question = 0
                    # print(iadx)
                    # For each question
                    traj_pool = [[] for _ in range(num_foresight)]
                    step_pool = [[] for _ in range(num_foresight)]
                    prob_pool = [[] for _ in range(num_foresight+1)]
                    adv_pool = [[] for _ in range(num_foresight+1)]
                    if "reclor" in args.data_path or "logiqa" in args.data_path:
                        traj_info = {'question_idx':i, 'question':'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}", 'ground_truth':test_data[i]['label'], 'foresight_part':[], 'final_part':{}}
                    else:
                        traj_info = {'question_idx':i, 'question':test_data[i]['input'], 'ground_truth':test_data[i]['target'], 'foresight_part':[], 'final_part':{}}
                    all_token_num = 0

                    # Record for final step selection
                    all_cumulative_logprob_list = [[] for _ in range(step_beam_size)]
                    token_num_list = [[] for _ in range(step_beam_size)]
                    
                    sample_id_pool = []
                    
                    traj_complete = False
                    previous_steps_list = ["The reasoning steps are:\n\n" for _ in range(step_beam_size)]
                    previous_q_value_list = [0.0 for _ in range(step_beam_size)]
                    each_step_prob_pool = [[] for _ in range(step_beam_size)]
                    T = 0
                    for T in range(num_foresight):
                        # For each question foresight times, after reaching foresight times, complete the rest directly
                        reasoning_steps_list = previous_steps_list

                        cur_foresight_info = {'foresight_epoch':T, 'steps':[], 'foresight_steps':[], 'cluster_info':{}} # For recording traj info

                        if "gsm" in args.data_path or "math" in args.data_path:
                            question = test_data[i]['input']
                            chat = [
                                {'role': 'system', 'content': system_prompt},
                                {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
                                {'role': 'assistant', 'content': ''}
                            ]
                        elif "aime" in args.data_path:
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
                        # input('inputs 1')
                        # print(inputs)
                        # print(inputs)
                        inputs = inputs.replace(stop_token, "").strip()
                        # input('inputs 1')
                        # print(inputs)
                        inputs_list = [inputs + reasoning_steps_list[beam_idx] for beam_idx in range(step_beam_size)] # 对每个step beam size进行
                        # 记录一下输入的tokens数量
                        print(inputs_list)
                        for each_input in inputs_list:
                            input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
                        sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["\n", "<end_of_reasoning>"])
                        # sampling_params = SamplingParams(max_tokens=128, n=num_rollout, logprobs=0, temperature=0.6)

                        # 根据当前的状态，foresight
                        outputs = model.generate(inputs_list, sampling_params)
                        total_rollout_times += num_rollout*step_beam_size
                        # 如果rollout的step的prob都很高且方差不大，就可以不用foresight，直接随机采
                        all_avg_logp = []
                        selected_steps = []
                        # inputs_list = []
                        candidates_list = []
                        normalized_logp_list = []
                        cumulative_logprob_list = []
                        output_token_num_list = []
                        aaa = [] # 这个变量只有下面那一个用处，就是帮助记录选择的那一步所用的logprob，ctrl f 找一下那一步就懂了
                        aaa_text = []
                        directly_sample_flag = False
                        for ii in range(step_beam_size):
                            for jj in range(num_rollout):
                                output = outputs[ii].outputs[jj]
                                response = output.text.strip()
                                aaa_text.append(response)
                                selected_steps.append(response)
                                candidates_list.append(response)
                                all_avg_logp.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                                aaa.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                                cumulative_logprob_list.append(output.cumulative_logprob)
                                output_token_num_list.append(len(output.token_ids))
                                all_token_num += len(output.token_ids)
                                # 给traj info记录信息
                                tem_step_info = {'index':ii*num_rollout+jj, 'text':response, 
                                                'adv':output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[ii]
                                                ,'logp':output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                                'cumulative_logp':output.cumulative_logprob ,'token_num':len(output.token_ids)}
                                cur_foresight_info['steps'].append(tem_step_info)
                                # 记录输出token数量
                                output_token_num_for_this_question += len(output.token_ids)
                                
                        
                        all_avg_logp = np.array(all_avg_logp)

                        # 选出需要继续foresight的idx
                        keep_foresight_list = []
                            
                        if args.pruning_strategy == "none" or args.pruning_strategy == "":
                            keep_foresight_list = list(range(step_beam_size*num_rollout))
                        else:
                            normalized_logp_list = []
                            adv_list = []
                            not_foresight_list = []
                            num_to_foresight = 0
                            not_foresight_normalized_logp_list = []
                            not_foresight_adv_list = []
                            for beam_idx in range(step_beam_size):
                                for j in range(num_rollout):
                                    output = outputs[beam_idx].outputs[j]
                                    normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                                    adv_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[beam_idx])
                            
                            # 计算normalized_logp_list的均值
                            if args.pruning_strategy == "sirno_low_sigma":
                                mean = np.mean(normalized_logp_list)
                                std = np.std(normalized_logp_list)

                                for iidx, each_logp in enumerate(normalized_logp_list):
                                    if each_logp > mean - args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_normalized_logp_list.append(each_logp)
                                
                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_normalized_logp_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()
                                    
                            elif args.pruning_strategy == "sirno_high_sigma":
                                mean = np.mean(normalized_logp_list)
                                std = np.std(normalized_logp_list)

                                for iidx, each_logp in enumerate(normalized_logp_list):
                                    if each_logp > mean + args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_normalized_logp_list.append(each_logp)
                                
                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_normalized_logp_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()
                            elif args.pruning_strategy == "sirno_both_sigma":
                                mean = np.mean(normalized_logp_list)
                                std = np.std(normalized_logp_list)

                                for iidx, each_logp in enumerate(normalized_logp_list):
                                    if abs(each_logp - mean) < args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_normalized_logp_list.append(each_logp)
                                
                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_normalized_logp_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_low_sigma':
                                mean = np.mean(adv_list)
                                std = np.std(adv_list)

                                for iidx, each_adv in enumerate(adv_list):
                                    if each_adv > mean - args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight
                                    
                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_above_avg':
                                for iidx, each_adv in enumerate(adv_list):
                                    if each_adv > 0:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_high_sigma':
                                mean = np.mean(adv_list)
                                std = np.std(adv_list)

                                for iidx, each_adv in enumerate(adv_list):
                                    if each_adv > mean + args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()

                            elif args.pruning_strategy == 'advno_both_sigma':
                                mean = np.mean(adv_list)
                                std = np.std(adv_list)

                                for iidx, each_adv in enumerate(adv_list):
                                    if abs(each_adv-mean) < args.sigma_rate * std:
                                        keep_foresight_list.append(iidx)
                                        num_to_foresight += 1
                                    else:
                                        not_foresight_list.append(iidx)
                                        not_foresight_adv_list.append(each_adv)

                                if num_to_foresight < step_beam_size:
                                    # 需要补充至少够step_beam_size个
                                    temp = 0.1
                                    num_to_add = step_beam_size - num_to_foresight

                                    weights = softmax([logp/temp for logp in not_foresight_adv_list])
                                    added_from_not_to_foresight_list = np.random.choice(len(weights), p=weights, size=num_to_add, replace=False).tolist()
                                    add_idx = [not_foresight_list[i] for i in added_from_not_to_foresight_list]
                                    keep_foresight_list += add_idx
                                    keep_foresight_list.sort()
                            
                                
                        print('本次foresight ', len(keep_foresight_list))
                        keep_foresight_list.sort()
                        total_rollout_times += len(keep_foresight_list) # 这里加的是后面foresight的次数
                        saved_rollout_times += (step_beam_size*num_rollout - len(keep_foresight_list))
                        inputs_list = []
                        candidates_list = ['' for _ in range(len(keep_foresight_list))]  
                        for idx, foresight_idx in enumerate(keep_foresight_list):
                            output = outputs[foresight_idx//num_rollout].outputs[foresight_idx%num_rollout]
                            response = output.text.strip()
                            reasoning_steps_candidate = reasoning_steps_list[foresight_idx//num_rollout] + "\n" + response
                            candidates_list[idx] = response

                            if "gsm" in args.data_path or "math" in args.data_path:
                                question = test_data[i]['input']
                                chat = [
                                    {'role': 'system', 'content': system_prompt},
                                    {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly output the reasoning steps.\n'},
                                    {'role': 'assistant', 'content': ''}
                                ]
                            elif "aime" in args.data_path:
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
                            inputs_list.append(tokenizer.apply_chat_template(
                                chat,
                                tokenize=False,
                            ).rstrip(stop_token).rstrip() + reasoning_steps_candidate)
                        # input('inputs 2')
                        # print(inputs_list[0])
                        # 把每一步进行补全
                        sampling_params = SamplingParams(max_tokens=1024 ,n=1, logprobs=1, stop="<end_of_reasoning>")
                        # sampling_params = SamplingParams(max_tokens=64 ,n=1, logprobs=1)

                        # 记录输入token数量
                        for each_input in inputs_list:
                            input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
                        outputs = model.generate(inputs_list, sampling_params)

                        # For selection
                        normalized_logp_list = []
                        advantages_list = []
                        output_text_list = []
                        for jf in range(len(inputs_list)):
                            output = outputs[jf].outputs[0]
                            response = output.text.strip()
                            normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                            advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[keep_foresight_list[jf]//num_rollout])
                            output_text_list.append(response)
                            all_token_num += len(output.token_ids)
                            # Record information for traj info
                            tem_foresight_info = {'index':jf, 'idx_in_origin':keep_foresight_list[jf], 'text':response,
                                                'adv':output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[keep_foresight_list[jf]//num_rollout],
                                                'logp':output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                                'cumulative_logp':output.cumulative_logprob ,'token_num':len(output.token_ids)}
                            cur_foresight_info['foresight_steps'].append(tem_foresight_info)
                            # Record output token count
                            output_token_num_for_this_question += len(output.token_ids)

                        # Select index through different strategies
                        if args.strategy == "highest":  # select the step with highest rewards
                            selected_index = normalized_logp_list.index(max(normalized_logp_list))
                        elif args.strategy == "random":  # select the step with random rewards
                            selected_index = random.randint(0, num_rollout-1)
                        elif args.strategy == "sir":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()
                        elif args.strategy == "sir_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                        elif args.strategy == "adv":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size).tolist()
                        elif args.strategy == "adv_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                        
                        elif args.strategy == 'cluster':
                            # Mask steps with adv < 0, while recording their index in the original list. If count is less than step_beam_size, randomly supplement
                            tem_output_text_list = []
                            tem_selected_index_list = []
                            tem_cluster_info = {}
                            if args.adv_above0 == "True":
                                mask = [adv > 0 for adv in advantages_list] # len(advantages_list) == len（keep_foresight_list）
                                for ddi in range(len(mask)): # ddi has no specific meaning, just represents index
                                    if mask[ddi] and output_text_list[ddi] != '': # Require adv > 0 and text not empty
                                        tem_selected_index_list.append(ddi)
                                tem_selected_index_list.sort()
                            else:
                                for dadagad in range(len(output_text_list)): # Remove empty strings
                                    if output_text_list[dadagad] != '':
                                        tem_selected_index_list.append(dadagad)

                            
                            if len(tem_selected_index_list) < step_beam_size:
                                tem_cluster_info['state'] = 'cannot cluster'
                                print('cannot cluster, use adv no replace')
                                weights = softmax([adv/temp for adv in advantages_list])
                                selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                                # print('selected_idx_list:', selected_index_list)
                            else:
                                try:
                                    # print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
                                    # len(output_text_list) == len(advantages_list) == len(keep_foresight_list)
                                    tem_output_text_list = [output_text_list[ddi] for ddi in tem_selected_index_list] # Elements in tem_selected_index_list are indices in keep_foresight_list
                                    tem_advantages_list = [advantages_list[ddi] for ddi in tem_selected_index_list]
                                    vectorizer =TfidfVectorizer()
                                    X=vectorizer.fit_transform(tem_output_text_list)
                                    k = args.cluster_num 
                                    kmeans = KMeans(n_clusters=k)
                                    kmeans.fit(X)
                                    cluster_labels = kmeans.labels_
                                    cluster_list = [[] for _ in range(k)]

                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list[cluster_label].append(aidx)
                                    cluster_list = [sorted(cluster) for cluster in cluster_list]

                                    cluster_len_ratio = [len(cluster)/len(tem_selected_index_list) for cluster in cluster_list]
                                    per_sample_cluster_len_ratio = [cluster_len_ratio[cluster_labels[ddi]] for ddi in range(len(tem_selected_index_list))]
                                    cluster_weights = softmax(per_sample_cluster_len_ratio)
                                    adv_weights = softmax([adv/temp for adv in tem_advantages_list])
                                    weights = [cluster_weights[ddi] + adv_weights[ddi] for ddi in range(len(tem_selected_index_list))]
                                    weights = [each_weight/2 for each_weight in weights]
                                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                                    selected_index_list = [tem_selected_index_list[ddi] for ddi in selected_index_list]
                                    # For recording traj
                                    cluster_list_foresight_idx = [[] for _ in range(k)] # Indices in keep_foresight_list
                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list_foresight_idx[cluster_label].append(tem_selected_index_list[aidx])
                                    cluster_list_origin_idx = [[keep_foresight_list[foresight_idx] for foresight_idx in fcluster] for fcluster in cluster_list_foresight_idx]
                                    tem_cluster_info['state'] = 'success'
                                    tem_cluster_info['cluster_result_cluster_idx'] = cluster_list
                                    tem_cluster_info['cluster_result_foresight_idx'] = cluster_list_foresight_idx
                                    tem_cluster_info['cluster_result_origin_idx'] = cluster_list_origin_idx
                                    
                                    # Determine whether to early stop
                                    threshold = float(args.threshold)
                                    for cluster_idx, ddcluster in enumerate(cluster_list):
                                        if len(cluster_list[cluster_idx])/len(tem_selected_index_list) > threshold:
                                            tem_cluster_info['early_stop'] = 'True'
                                            stop_foresight = True
                                            break
                                except:
                                    tem_cluster_info['state'] =  'fail'
                                    print('cannot cluster, use adv no replace')
                                    weights = softmax([adv/temp for adv in advantages_list])
                                    selected_index_list = np.random.choice(len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                            cur_foresight_info['cluster_info'] = tem_cluster_info
                            

                        cur_foresight_info['selected_idx_in_foresight'] = selected_index_list
                        cur_foresight_info['selected_idx_in_origin'] = [keep_foresight_list[iiiidx] for iiiidx in selected_index_list]

                        traj_info['foresight_part'].append(cur_foresight_info)

                        sample_id_pool.append([keep_foresight_list[iiiidx] for iiiidx in selected_index_list]) # Selected idx refers to idx in keep_foresight_list
                        # Add to each_step_prob_pool
                        tem_logprob_list = [[] for _ in range(step_beam_size)]
                        for iii in range(step_beam_size):
                            tem_logprob_list[iii] = all_cumulative_logprob_list[iii]

                                        
                        
                        previous_steps_list_updated, previous_q_value_list = [], []
                        for m, selected_index in enumerate(selected_index_list):
                            previous_steps_list_updated.append(previous_steps_list[keep_foresight_list[selected_index]//num_rollout] + candidates_list[selected_index].strip() + "\n") # Note that the length of candidates_list is not num_rollout*step_beam_size
                            previous_q_value_list.append(normalized_logp_list[selected_index])

                        previous_steps_list = previous_steps_list_updated

                        # Used to determine whether to continue foresight, storing data from this foresight
                        selected_cumulative_logprob_list = []
                        idx_in_origin_list = [keep_foresight_list[iiia] for iiia in selected_index_list]
                        text_for_early_stop = []
                        for jjjj in range(step_beam_size):
                            each_step_prob_pool[jjjj].append(aaa[idx_in_origin_list[jjjj]])
                            all_cumulative_logprob_list[jjjj] = tem_logprob_list[idx_in_origin_list[jjjj]//num_rollout] + [cumulative_logprob_list[idx_in_origin_list[jjjj]]]      
                            token_num_list[jjjj] = token_num_list[idx_in_origin_list[jjjj]//num_rollout] + [output_token_num_list[idx_in_origin_list[jjjj]]]   
                            selected_cumulative_logprob_list.append(cumulative_logprob_list[idx_in_origin_list[jjjj]])       
                            text_for_early_stop.append(aaa_text[idx_in_origin_list[jjjj]])

                        


                        if args.depth_pruning_strategy != "none" and args.depth_pruning_strategy != "":
                            # If all steps are the same, stop directly
                            just_stop = True
                            for each_text in text_for_early_stop:
                                if each_text == text_for_early_stop[0]:
                                    continue
                                else:
                                    just_stop = False
                            if just_stop and T>= args.least_foresight_num:
                                print('question ', i, ' just stop at depth (all the same)', T)
                                # print(text_for_early_stop)
                                break

                            if args.depth_pruning_strategy == "cluster":
                                if stop_foresight and T+1 >= args.least_foresight_num: #    At least foresight times
                                    print('question ', i, ' stop foresight at depth ', T)
                                    break
                    
                    if "gsm" in args.data_path or "math" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly output the reasoning steps.\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "aime" in args.data_path:
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
                    # Record input token count
                    for each_input in inputs_list:
                        input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])
                    outputs = model.generate(inputs_list, sampling_params)
                    total_rollout_times += step_beam_size
                    
                    candidates_list = []
                    normalized_logp_list = []
                    advantages_list = []
                    answer_avg_logprob_list = []
                    output_text_list = []
                    final_answer_info = []
                    for jx in range(step_beam_size):
                        output = outputs[jx].outputs[0]
                        response = output.text.strip()
                        candidates_list.append(response)
                        normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
                        advantages_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx])
                        all_cumulative_logprob_list[jx].append(output.cumulative_logprob)
                        token_num_list[jx].append(len(output.token_ids))
                        output_text_list.append(response)
                        all_token_num += len(output.token_ids)
                        # Record information for traj info
                        tem_answer_info = {'index':jx, 'text':response, 'adv':output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx],
                                        'logp':output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                        'cumulative_logp':output.cumulative_logprob ,'token_num':len(output.token_ids)}
                        final_answer_info.append(tem_answer_info)
                        # Record output token count
                        output_token_num_for_this_question += len(output.token_ids)
                    traj_info['final_part']['final_answer'] = final_answer_info
                    # Now only select one index
                    if args.final_select_strategy == "same_as_strategy":
                        if args.strategy == "cluster":
                            # Mask steps with adv < 0
                            tem_output_text_list = []
                            tem_selected_index_list = []
                            cur_cluster_info = {}
                            if args.adv_above0 == "True":
                                mask = [adv > 0 for adv in advantages_list] # len(advantages_list) == len（keep_foresight_list）
                                for ddi in range(len(mask)): # ddi has no specific meaning, just represents index
                                    if mask[ddi] and output_text_list[ddi] != '': # Require adv > 0 and text not empty
                                        tem_selected_index_list.append(ddi)
                            else:
                                for dadagad in range(len(output_text_list)): # Remove empty strings
                                    if output_text_list[dadagad] != '':
                                        tem_selected_index_list.append(dadagad)

                            if len(tem_selected_index_list) == 0:
                                # Use adv no replace
                                cur_cluster_info['state'] = 'fail'
                                weights = softmax([-adv/temp for adv in advantages_list]) # NOTE: Added negative sign here
                                selected_index_final = np.random.choice(len(weights), p=weights)
                            else:
                                try:
                                    # The following part is to cluster into two classes and choose one from the larger class
                                    tem_output_text_list = [output_text_list[ddi] for ddi in tem_selected_index_list] # Elements in tem_selected_index_list are indices in keep_foresight_list
                                    tem_advantages_list = [advantages_list[ddi] for ddi in tem_selected_index_list]
                                    vectorizer =TfidfVectorizer()
                                    X=vectorizer.fit_transform(tem_output_text_list)
                                    k = 2 # TODO: may change later
                                    kmeans = KMeans(n_clusters=k)
                                    kmeans.fit(X)
                                    cluster_labels = kmeans.labels_
                                    cluster_list = [[] for _ in range(k)]

                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        cluster_list[cluster_label].append(aidx)
                                    cluster_list = [sorted(cluster) for cluster in cluster_list]
                                    cluster0 = []
                                    cluster1 = []
                                    for aidx, cluster_label in enumerate(cluster_labels):
                                        if cluster_label == 0:
                                            cluster0.append(aidx)
                                        else:
                                            cluster1.append(aidx)
                                    if len(cluster0) > len(cluster1):
                                        cluster_adv_list = [tem_advantages_list[ddi] for ddi in cluster0]
                                        weights = softmax([adv/temp for adv in cluster_adv_list])
                                        selected_index_in_cluster = np.random.choice(len(weights), p=weights)
                                        selected_index_in_tem = cluster0[selected_index_in_cluster]
                                        selected_index_final = tem_selected_index_list[selected_index_in_tem]
                                    else:
                                        cluster_adv_list = [tem_advantages_list[ddi] for ddi in cluster1]
                                        weights = softmax([adv/temp for adv in cluster_adv_list])
                                        selected_index_in_cluster = np.random.choice(len(weights), p=weights)
                                        selected_index_in_tem = cluster0[selected_index_in_cluster]
                                        selected_index_final = tem_selected_index_list[selected_index_in_tem]
                                    cur_cluster_info['state'] = 'success'
                                    cur_cluster_info['cluster_result_cluster_idx'] = cluster_list
                                    cur_cluster_info['cluster_result_answer_idx'] = [[tem_selected_index_list[iodx] for iodx in cluster] for cluster in cluster_list]
                                except:
                                    # Use adv no replace
                                    cur_cluster_info['state'] = 'fail'
                                    weights = softmax([adv/temp for adv in advantages_list])
                                    selected_index_final = np.random.choice(len(weights), p=weights)
                                cur_cluster_info['selected_idx_in_origin'] = selected_index_final
                            traj_info['final_part']['cluster'] = cur_cluster_info
                            # traj_info['selected_idx_in_origin'] = [keep_foresight_list[iiiidx] for iiiidx in selected_index_list]

                        if args.strategy == "sir":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_final = np.random.choice(len(weights), p=weights)
                        if args.strategy == "sir_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in normalized_logp_list])
                            selected_index_final = np.random.choice(len(weights), p=weights, replace=False)
                        elif args.strategy == "adv":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_final = np.random.choice(len(weights), p=weights)
                        elif args.strategy == "adv_no_replace":
                            temp = 0.1
                            def softmax(x):
                                e_x = np.exp(np.array(x))
                                return e_x / e_x.sum(axis=0)
                            weights = softmax([logp/temp for logp in advantages_list])
                            selected_index_final = np.random.choice(len(weights), p=weights, replace=False)
                    else:
                        if args.final_select_strategy == "max_logprob":
                            max_idx = 0
                            
                            max_logprob_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                # 找到step_beam_size个最大的logprob
                                tem_max = -INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob > tem_max:
                                        tem_max = each_step_logprob
                                max_logprob_pool.append(tem_max)
                            
                            # 从max_logprob_pool中找到最大的
                            max_logprob = -INF
                            for idx, each_max_logprob in enumerate(max_logprob_pool):
                                if each_max_logprob > max_logprob:
                                    max_logprob = each_max_logprob
                                    max_idx = idx
                            
                            selected_index_final = max_idx

                        
                        elif args.final_select_strategy == "min_logprob":
                            min_idx = 0
                            
                            min_logprob_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                # 找到step_beam_size个最小的logprob
                                tem_min = INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob < tem_min:
                                        tem_min = each_step_logprob
                                min_logprob_pool.append(tem_min)
                            
                            # 从max_logprob_pool中找到最大的
                            min_logprob = -INF
                            for idx, each_min_logprob in enumerate(min_logprob_pool):
                                if each_min_logprob > min_logprob:
                                    min_logprob = each_min_logprob
                                    min_idx = idx
                            
                            selected_index_final = min_idx

                        elif args.final_select_strategy == "max_min_diff":
                            max_min_diff_idx = 0
                            
                            max_min_diff_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                # 找到step_beam_size个最大的logprob
                                tem_max = -INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob > tem_max:
                                        tem_max = each_step_logprob
                                
                                # 找到step_beam_size个最小的logprob
                                tem_min = INF
                                for each_step_logprob in each_step_prob_list:
                                    if each_step_logprob < tem_min:
                                        tem_min = each_step_logprob
                                
                                max_min_diff_pool.append(tem_max - tem_min)
                            
                            # 从max_logprob_pool中找到最大的
                            max_min_diff = -INF
                            for idx, each_max_min_diff in enumerate(max_min_diff_pool):
                                if each_max_min_diff > max_min_diff:
                                    max_min_diff = each_max_min_diff
                                    max_min_diff_idx = idx
                            
                            selected_index_final = max_min_diff_idx

                        elif args.final_select_strategy == "step_avg":
                            # 把每一步的avg再avg
                            avg_idx = 0
                            
                            avg_logprob_pool = []
                            for idx, each_step_prob_list in enumerate(each_step_prob_pool):
                                tem_avg = sum(each_step_prob_list) / len(each_step_prob_list)
                                avg_logprob_pool.append(tem_avg)
                            
                            avg_logprob = -INF
                            for idx, each_avg_logprob in enumerate(avg_logprob_pool):
                                if each_avg_logprob > avg_logprob:
                                    avg_logprob = each_avg_logprob
                                    avg_idx = idx
                            
                            selected_index_final = avg_idx

                        elif args.final_select_strategy == "sum_avg":
                            # 把所有的求和之后再avg
                            sum_avg_idx = 0

                            sum_avg_logprob_pool = []
                            for idx, each_cumulative_prob_list in enumerate(all_cumulative_logprob_list):
                                tem_sum = sum(each_cumulative_prob_list)
                                token_num = sum(token_num_list[idx])
                                sum_avg_logprob_pool.append(tem_sum/(token_num+1e-8))

                            sum_avg_logprob = -INF
                            for idx, each_sum_avg_logprob in enumerate(sum_avg_logprob_pool):
                                if each_sum_avg_logprob > sum_avg_logprob:
                                    sum_avg_logprob = each_sum_avg_logprob
                                    sum_avg_idx = idx

                            selected_index_final = sum_avg_idx
                    break
                # except:
                #     try_time += 1
                #     if try_time == 3:
                #         with open('/cpfs01/user/xufangzhi/o1/cluster_results/over_length_results/'+ffname+'.txt', 'a') as fsa:
                #             fsa.write(str(i))
                #             fsa.write('\n')


            sample_id_pool.append([selected_index_final for _ in range(step_beam_size)])
            whole_traj = previous_steps_list[selected_index_final] + "\n" + candidates_list[selected_index_final] # 最终答案的traj
            whole_traj_list = [previous_steps_list[beam_idx] + "\n" + candidates_list[beam_idx] for beam_idx in range(step_beam_size)] # 每个beam的traj
            traj_info['token_num'] = all_token_num
            all_traj_info.append(traj_info)
            
            ###########################################
            #           Write to result file          #
            ###########################################
            result = {}
            result['id'] = i
            if "gsm" in args.data_path or "math" in args.data_path or "aime" in args.data_path:
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

            
            if args.record_process:
                with open(args.time_path+"TRAJ_INFO-"+ffname+'.json', 'w') as fa:
                    fa.write(json.dumps(all_traj_info) + '\n')
            f.write(json.dumps(result) + '\n')
            f.flush()
            all_output_token_num += output_token_num_for_this_question
            all_input_token_num += input_token_num_for_this_question
            print('output_token_num_for_this_question: ', output_token_num_for_this_question)
            print('input_token_num_for_this_question: ', input_token_num_for_this_question)
            print('all_output_token_num: ', all_output_token_num)
            print('all_input_token_num: ', all_input_token_num)
    end_time = time.time()
    time_span = end_time - start_time
    print(f"time: {time_span}")
    # ffname = 'sigmarate_' + args.sigma_rate + '-' +  args.pruning_strategy
    time_path = args.time_path + ffname +  '.txt'
    with open(time_path, 'a') as f:
        f.write('time:  ' + str(time_span) + '\n')
        f.write('total:  ' + str(total_rollout_times) + '\n')
        f.write('save:  ' + str(saved_rollout_times) + '\n')
        f.write('num_rollout:  ' + str(num_rollout) + '\n')
        f.write('num_foresight:  ' + str(num_foresight) + '\n')
        f.write('step_beam_size:  ' + str(step_beam_size) + '\n')
        f.write('strategy:  ' + str(args.strategy) + '\n')
        f.write('final_select_strategy:  ' + str(args.final_select_strategy) + '\n')
        f.write('pruning_strategy:  ' + str(args.pruning_strategy) + '\n')
        f.write('depth_pruning_strategy:  ' + str(args.depth_pruning_strategy) + '\n')
        f.write('threshold:  ' + str(args.threshold) + '\n')
        f.write('sigma_rate:  ' + str(args.sigma_rate) + '\n')
        f.write('adv_above0:  ' + str(args.adv_above0) + '\n')
        f.write('cluster_num:  ' + str(args.cluster_num) + '\n')
        f.write('all_input_token_num:  ' + str(all_input_token_num) + '\n')
        f.write('all_output_token_num:  ' + str(all_output_token_num) + '\n')
    print('total rollouts: ', total_rollout_times)
    print('saved rollouts: ', saved_rollout_times)
    print('all_output_token_num: ', all_output_token_num)
    print('all_input_token_num: ', all_input_token_num)
