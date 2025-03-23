# this file contains the original code with a few modifications for readability(e.g.: comments, variable names, delete useless variables, etc.)
import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import (
    StoppingCriteriaList,
    StoppingCriteria,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from vllm import LLM, SamplingParams
from data.logic_example import LOGIC_MRC_COT_4_SHOT
from data.math_example import MATH_POT_FEW_SHOT, MATH_COT_FEW_SHOT, GSM_COT_8_SHOT, MATH_COT_4_SHOT
import argparse
import torch
import numpy as np
import random
import json
import os

INF = 10
temp = 0.1


def softmax(x):
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='llama3.1',
                        help="model id, please fill the exact model path in the code below")
    parser.add_argument('--data_path', type=str,
                        default='data/gsm_test.json', help='the test data path')
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed")
    parser.add_argument('--step_beam_size', type=int, default=4,
                        help="step beam size")
    parser.add_argument('--num_rollout', type=int, default=4,
                        help="number of rollout")
    parser.add_argument('--num_foresight', type=int, default=8,
                        help="number of foresight")
    parser.add_argument('--record_process', type=bool, default=True,
                        help="whether to record the whole process")
    parser.add_argument('--strategy', type=str, default='cluster',
                        help="strategy for selection, default is cluster")
    parser.add_argument('--time_path', type=str, default='./results/time/',
                        help="the path to save the time information and trajectory information of each question")
    parser.add_argument('--final_select_strategy',
                        type=str, default='same_as_strategy', help="no need to change")
    parser.add_argument('--width_pruning_strategy', type=str,
                        default='low_sigma', help="width pruning strategy")
    parser.add_argument('--sigma_rate', type=float, default=1.0,
                        help="sigma rate for width pruning")
    parser.add_argument('--depth_pruning_strategy',
                        type=str, default='cluster', help="depth pruning strategy")
    parser.add_argument('--threshold', type=float, default=0.7,
                        help="threshold for early stopping")
    parser.add_argument('--least_foresight_num', type=int, default=4,
                        help="least foresight number")
    parser.add_argument('--adv_above0', type=str, default='True',
                        help="only keep the steps with adv > 0")
    parser.add_argument('--file_name', type=str, default='250215-1',
                        help="file name")
    parser.add_argument('--cluster_num', type=int, default=3,
                        help="cluster number")
    parser.add_argument('--gpus', type=int, default=2,
                        help="number of gpus")

    args = parser.parse_args()
    start_time = time.time()
    np.random.seed(args.seed)
    ffname = args.file_name
    args.output_path = args.output_dir + ffname + '.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.exists(args.time_path):
        os.makedirs(args.time_path)
    if args.model_id == "llama3.1":
        PATH_TO_CONVERTED_WEIGHTS = ""
    elif args.model_id == "qwen3B":
        PATH_TO_CONVERTED_WEIGHTS = ""
    elif args.model_id == "mistral":
        PATH_TO_CONVERTED_WEIGHTS = ""
    elif args.model_id == "llama70B":
        PATH_TO_CONVERTED_WEIGHTS = ""

    tokenizer = AutoTokenizer.from_pretrained(
        PATH_TO_CONVERTED_WEIGHTS, max_length=32768, trust_remote_code=True)

    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    stop_token = tokenizer.eos_token

    saved_rollout_times = 0
    total_rollout_times = 0
    model = LLM(model=PATH_TO_CONVERTED_WEIGHTS, tensor_parallel_size=args.gpus,
                max_model_len=32768, trust_remote_code=True)

    num_rollout = args.num_rollout
    num_foresight = args.num_foresight
    step_beam_size = args.step_beam_size

    DATA_PATH = args.data_path
    with open(DATA_PATH) as file:
        test_data = json.load(file)

    OUTPUT_PATH = args.output_path

    if "gsm" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}"
    elif "math" in args.data_path or "aime" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}"
    elif "reclor" in args.data_path or "logiqa" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Please finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{LOGIC_MRC_COT_4_SHOT}\n"
    elif "strategy" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with one question. At the end, you must output 'Yes' or 'No' after 'The answer is: '."
    elif "gpqa" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response."
    elif "arc" in args.data_path:
        system_prompt = f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response."

    all_output_token_num = 0  # record the number of output tokens during the decoding
    all_input_token_num = 0  # record the number of input tokens during the decoding
    with open(OUTPUT_PATH, "w") as f:
        print('len test data', len(test_data))
        all_traj_info = []

        for i in range(len(test_data)):
            try_time = 0
            stop_foresight = False
            output_token_num_for_this_question = 0
            input_token_num_for_this_question = 0

            if "reclor" in args.data_path or "logiqa" in args.data_path:
                traj_info = {'question_idx': i, 'question': 'Passage: ' + test_data[i]['context'] + '\nQuestion: ' + test_data[i]['question'] +
                             f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}", 'ground_truth': test_data[i]['label'], 'foresight_part': [], 'final_part': {}}
            else:
                traj_info = {'question_idx': i, 'question': test_data[i]['input'], 'ground_truth': test_data[i]['target'], 'foresight_part': [
                ], 'final_part': {}}
            all_token_num = 0

            all_cumulative_logprob_list = [[] for _ in range(step_beam_size)]
            token_num_list = [[] for _ in range(step_beam_size)]

            previous_steps_list = [
                "The reasoning steps are:\n\n" for _ in range(step_beam_size)]
            previous_q_value_list = [0.0 for _ in range(step_beam_size)]
            each_step_prob_pool = [[] for _ in range(step_beam_size)]
            T = 0
            for T in range(num_foresight):

                ###################### generate the candidate steps (begin)######################

                reasoning_steps_list = previous_steps_list

                cur_foresight_info = {'foresight_epoch': T, 'steps': [], 'foresight_steps': [
                ], 'cluster_info': {}}  # Used to record traj information

                if "gsm" in args.data_path or "math" in args.data_path:
                    question = test_data[i]['input']
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'The question: ' + question +
                            '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "reclor" in args.data_path or "logiqa" in args.data_path:
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: ' + test_data[i]['question'] +
                            f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "gpqa" in args.data_path:
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'Passage: ' +
                            test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                        {'role': 'assistant', 'content': ''}
                    ]
                elif "arc" in args.data_path:
                    chat = [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': 'Passage: ' +
                            test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                        {'role': 'assistant', 'content': ''}
                    ]
                if args.model_id == "mistral":
                    chat[1]['content'] = system_prompt + \
                        "\n" + chat[1]['content']
                    chat = chat[1:]

                inputs = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                )
                inputs = inputs.replace(stop_token, "").strip()

                inputs_list = [inputs + reasoning_steps_list[beam_idx]
                               for beam_idx in range(step_beam_size)]  # For each step beam size
                # Record the number of input tokens
                for each_input in inputs_list:
                    input_token_num_for_this_question += len(
                        tokenizer(each_input)['input_ids'])
                sampling_params = SamplingParams(
                    max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["\n", "<end_of_reasoning>"])

                outputs = model.generate(inputs_list, sampling_params)
                total_rollout_times += num_rollout*step_beam_size
                ###################### generate the candidate steps (end)######################

                ###################### in width pruning (begin)######################
                cumulative_logprob_list = []
                output_token_num_list = []
                aaa = []  # This variable is only used below to help record the logprob used for the selected step, ctrl f to find that step
                aaa_text = []
                for ii in range(step_beam_size):
                    for jj in range(num_rollout):
                        output = outputs[ii].outputs[jj]
                        response = output.text.strip()
                        aaa_text.append(response)
                        aaa.append(output.cumulative_logprob /
                                   (len(output.token_ids)+1e-8))
                        cumulative_logprob_list.append(
                            output.cumulative_logprob)
                        output_token_num_list.append(len(output.token_ids))
                        all_token_num += len(output.token_ids)
                        # Record information for traj info
                        tem_step_info = {'index': ii*num_rollout+jj, 'text': response,
                                         'adv': output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[ii], 'logp': output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                         'cumulative_logp': output.cumulative_logprob, 'token_num': len(output.token_ids)}
                        cur_foresight_info['steps'].append(tem_step_info)
                        # Record the number of output tokens
                        output_token_num_for_this_question += len(
                            output.token_ids)

                # Select the idx to continue foresight
                keep_foresight_list = []

                if args.width_pruning_strategy == "none" or args.width_pruning_strategy == "":
                    keep_foresight_list = list(
                        range(step_beam_size*num_rollout))
                else:
                    normalized_logp_list = []
                    not_foresight_list = []
                    num_to_foresight = 0
                    not_foresight_normalized_logp_list = []
                    for beam_idx in range(step_beam_size):
                        for j in range(num_rollout):
                            output = outputs[beam_idx].outputs[j]
                            normalized_logp_list.append(
                                output.cumulative_logprob / (len(output.token_ids)+1e-8))
                    if args.width_pruning_strategy == "low_sigma":
                        mean = np.mean(normalized_logp_list)
                        std = np.std(normalized_logp_list)

                        for iidx, each_logp in enumerate(normalized_logp_list):
                            if each_logp > mean - args.sigma_rate * std:
                                keep_foresight_list.append(iidx)
                                num_to_foresight += 1
                            else:
                                not_foresight_list.append(iidx)
                                not_foresight_normalized_logp_list.append(
                                    each_logp)

                        if num_to_foresight < step_beam_size:
                            # Need to supplement at least enough step_beam_size
                            temp = 0.1
                            num_to_add = step_beam_size - num_to_foresight

                            weights = softmax(
                                [logp/temp for logp in not_foresight_normalized_logp_list])
                            added_from_not_to_foresight_list = np.random.choice(
                                len(weights), p=weights, size=num_to_add, replace=False).tolist()
                            add_idx = [not_foresight_list[i]
                                       for i in added_from_not_to_foresight_list]
                            keep_foresight_list += add_idx
                            keep_foresight_list.sort()

                print('This iteration foresight ', len(
                    keep_foresight_list), " paths")
                keep_foresight_list.sort()

                ###################### in width pruning (end)######################

                total_rollout_times += len(keep_foresight_list)
                saved_rollout_times += (step_beam_size *
                                        num_rollout - len(keep_foresight_list))

                ###################### foresight (begin)######################
                inputs_list = []
                candidates_list = ['' for _ in range(
                    len(keep_foresight_list))]
                for idx, foresight_idx in enumerate(keep_foresight_list):
                    output = outputs[foresight_idx //
                                     num_rollout].outputs[foresight_idx % num_rollout]
                    response = output.text.strip()
                    reasoning_steps_candidate = reasoning_steps_list[foresight_idx //
                                                                     num_rollout] + "\n" + response
                    candidates_list[idx] = response

                    if "gsm" in args.data_path or "math" in args.data_path:
                        question = test_data[i]['input']
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'The question: ' + question +
                                '\nPlease directly output the reasoning steps.\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "reclor" in args.data_path or "logiqa" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: ' + test_data[i]['question'] +
                                f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}\n"},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "gpqa" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' +
                                test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    elif "arc" in args.data_path:
                        chat = [
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': 'Passage: ' +
                                test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                            {'role': 'assistant', 'content': ''}
                        ]
                    if args.model_id == "mistral":
                        chat[1]['content'] = system_prompt + \
                            "\n" + chat[1]['content']
                        chat = chat[1:]
                    inputs_list.append(tokenizer.apply_chat_template(
                        chat,
                        tokenize=False,
                    ).rstrip(stop_token).rstrip() + reasoning_steps_candidate)

                # foresight
                sampling_params = SamplingParams(
                    max_tokens=1024, n=1, logprobs=1, stop="<end_of_reasoning>")

                # Record the number of input tokens
                for each_input in inputs_list:
                    input_token_num_for_this_question += len(
                        tokenizer(each_input)['input_ids'])
                outputs = model.generate(inputs_list, sampling_params)

                # Used for selection
                normalized_logp_list = []
                advantages_list = []
                output_text_list = []
                for jf in range(len(inputs_list)):
                    output = outputs[jf].outputs[0]
                    response = output.text.strip()
                    normalized_logp_list.append(
                        output.cumulative_logprob / (len(output.token_ids)+1e-8))
                    advantages_list.append(output.cumulative_logprob / (len(
                        output.token_ids)+1e-8) - previous_q_value_list[keep_foresight_list[jf]//num_rollout])
                    output_text_list.append(response)
                    all_token_num += len(output.token_ids)
                    # Record information for traj info
                    tem_foresight_info = {'index': jf, 'idx_in_origin': keep_foresight_list[jf], 'text': response,
                                          'adv': output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[keep_foresight_list[jf]//num_rollout],
                                          'logp': output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                          'cumulative_logp': output.cumulative_logprob, 'token_num': len(output.token_ids)}
                    cur_foresight_info['foresight_steps'].append(
                        tem_foresight_info)
                    # Record the number of output tokens
                    output_token_num_for_this_question += len(
                        output.token_ids)

                ###################### foresight (end)######################

                ###################### cluster & selection (begin)######################
                if args.strategy == 'cluster':
                    # mask steps with adv less than 0, and record their index in the original list. If the number is less than step_beam_size, randomly supplement
                    tem_output_text_list = []
                    tem_selected_index_list = []
                    tem_cluster_info = {}
                    if args.adv_above0 == "True":
                        mask = [adv > 0 for adv in advantages_list]
                        # ddi has no specific meaning, just represents the index
                        for ddi in range(len(mask)):
                            # Require adv greater than 0 and text not empty
                            if mask[ddi] and output_text_list[ddi] != '':
                                tem_selected_index_list.append(ddi)
                        tem_selected_index_list.sort()
                    else:
                        # Remove empty strings
                        for dadagad in range(len(output_text_list)):
                            if output_text_list[dadagad] != '':
                                tem_selected_index_list.append(dadagad)

                    if len(tem_selected_index_list) < step_beam_size:
                        tem_cluster_info['state'] = 'cannot cluster'
                        print(
                            'Due to the number of paths is less than step_beam_size, cannot cluster, use adv no replace')
                        weights = softmax(
                            [adv/temp for adv in advantages_list])
                        selected_index_list = np.random.choice(
                            len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                    else:
                        try:
                            # tem_selected_index_list elements are in keep_foresight_list
                            tem_output_text_list = [
                                output_text_list[ddi] for ddi in tem_selected_index_list]
                            tem_advantages_list = [
                                advantages_list[ddi] for ddi in tem_selected_index_list]
                            vectorizer = TfidfVectorizer()
                            X = vectorizer.fit_transform(
                                tem_output_text_list)
                            k = args.cluster_num  # TODO: may change later
                            kmeans = KMeans(n_clusters=k)
                            kmeans.fit(X)
                            cluster_labels = kmeans.labels_
                            cluster_list = [[] for _ in range(k)]

                            for aidx, cluster_label in enumerate(cluster_labels):
                                cluster_list[cluster_label].append(aidx)
                            cluster_list = [sorted(cluster)
                                            for cluster in cluster_list]

                            cluster_len_ratio = [
                                len(cluster)/len(tem_selected_index_list) for cluster in cluster_list]
                            per_sample_cluster_len_ratio = [
                                cluster_len_ratio[cluster_labels[ddi]] for ddi in range(len(tem_selected_index_list))]
                            cluster_weights = softmax(
                                per_sample_cluster_len_ratio)
                            adv_weights = softmax(
                                [adv/temp for adv in tem_advantages_list])
                            weights = [cluster_weights[ddi] + adv_weights[ddi]
                                       for ddi in range(len(tem_selected_index_list))]
                            weights = [each_weight /
                                       2 for each_weight in weights]
                            selected_index_list = np.random.choice(
                                len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                            selected_index_list = [
                                tem_selected_index_list[ddi] for ddi in selected_index_list]
                            # Index in keep_foresight_list
                            cluster_list_foresight_idx = [
                                [] for _ in range(k)]
                            for aidx, cluster_label in enumerate(cluster_labels):
                                cluster_list_foresight_idx[cluster_label].append(
                                    tem_selected_index_list[aidx])
                            cluster_list_origin_idx = [
                                [keep_foresight_list[foresight_idx] for foresight_idx in fcluster] for fcluster in cluster_list_foresight_idx]
                            tem_cluster_info['state'] = 'success'
                            tem_cluster_info['cluster_result_cluster_idx'] = cluster_list
                            tem_cluster_info['cluster_result_foresight_idx'] = cluster_list_foresight_idx
                            tem_cluster_info['cluster_result_origin_idx'] = cluster_list_origin_idx

                            # Determine whether to early stop
                            threshold = float(args.threshold)
                            for cluster_idx, ddcluster in enumerate(cluster_list):
                                if len(cluster_list[cluster_idx])/len(tem_selected_index_list) > threshold:
                                    tem_cluster_info['early_stop'] = 'True'
                                    stop_foresight = True  # used for depth pruning, early stop
                                    break
                        except:
                            tem_cluster_info['state'] = 'fail'
                            print('cannot cluster, use adv no replace')
                            weights = softmax(
                                [adv/temp for adv in advantages_list])
                            selected_index_list = np.random.choice(
                                len(weights), p=weights, size=step_beam_size, replace=False).tolist()
                    cur_foresight_info['cluster_info'] = tem_cluster_info

                else:
                    print('not valid strategy')
                    exit()

                cur_foresight_info['selected_idx_in_foresight'] = selected_index_list
                cur_foresight_info['selected_idx_in_origin'] = [
                    keep_foresight_list[iiiidx] for iiiidx in selected_index_list]

                traj_info['foresight_part'].append(cur_foresight_info)

                # Add things to each_step_prob_pool
                tem_logprob_list = [[] for _ in range(step_beam_size)]
                for iii in range(step_beam_size):
                    tem_logprob_list[iii] = all_cumulative_logprob_list[iii]

                previous_steps_list_updated, previous_q_value_list = [], []
                for m, selected_index in enumerate(selected_index_list):
                    # Note that the length of candidates_list here is not num_rollout*step_beam_size
                    previous_steps_list_updated.append(
                        previous_steps_list[keep_foresight_list[selected_index]//num_rollout] + candidates_list[selected_index].strip() + "\n")
                    previous_q_value_list.append(
                        normalized_logp_list[selected_index])

                previous_steps_list = previous_steps_list_updated

                # Used to determine whether to continue foresight, storing data from this foresight
                selected_cumulative_logprob_list = []
                idx_in_origin_list = [keep_foresight_list[iiia]
                                      for iiia in selected_index_list]
                text_for_early_stop = []
                for jjjj in range(step_beam_size):
                    each_step_prob_pool[jjjj].append(
                        aaa[idx_in_origin_list[jjjj]])
                    all_cumulative_logprob_list[jjjj] = tem_logprob_list[idx_in_origin_list[jjjj]//num_rollout] + [
                        cumulative_logprob_list[idx_in_origin_list[jjjj]]]
                    token_num_list[jjjj] = token_num_list[idx_in_origin_list[jjjj] //
                                                          num_rollout] + [output_token_num_list[idx_in_origin_list[jjjj]]]
                    selected_cumulative_logprob_list.append(
                        cumulative_logprob_list[idx_in_origin_list[jjjj]])
                    text_for_early_stop.append(
                        aaa_text[idx_in_origin_list[jjjj]])

                ###################### cluster & selection (end)######################

                ###################### depth pruning (begin)######################
                if args.depth_pruning_strategy != "none" and args.depth_pruning_strategy != "":
                    # All steps are the same, just stop
                    just_stop = True
                    for each_text in text_for_early_stop:
                        if each_text == text_for_early_stop[0]:
                            continue
                        else:
                            just_stop = False
                    if just_stop and T >= args.least_foresight_num:
                        print('question ', i,
                              ' just stop at depth (all the same)', T)
                        break  # in depth pruning, early stop

                    if args.depth_pruning_strategy == "cluster":
                        if stop_foresight and T+1 >= args.least_foresight_num:  # At least foresight times
                            print('question ', i,
                                  ' stop foresight at depth ', T)
                            break  # in depth pruning, early stop

                ###################### depth pruning (end)######################

            ###################### generation final answer (begin)######################
            if "gsm" in args.data_path or "math" in args.data_path:
                question = test_data[i]['input']
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'The question: ' + question +
                        '\nPlease directly output the reasoning steps.\n'},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "aime" in args.data_path:
                question = test_data[i]['input']
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'The question: ' + question +
                        '\nPlease directly output the reasoning steps.\n'},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Passage: ' + test_data[i]['context'] + '\nQuestion: ' + test_data[i]['question'] +
                        f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}\n"},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "gpqa" in args.data_path:
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Passage: ' +
                        test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                    {'role': 'assistant', 'content': ''}
                ]
            elif "arc" in args.data_path:
                chat = [
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': 'Passage: ' +
                        test_data[i]['input'] + '\n\nThe reasoning steps are:\n\n'},
                    {'role': 'assistant', 'content': ''}
                ]
            if args.model_id == "mistral":
                chat[1]['content'] = system_prompt + \
                    "\n" + chat[1]['content']
                chat = chat[1:]
            inputs = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True
            ).rstrip(stop_token).rstrip()

            inputs_list = [inputs + previous_steps_list[beam_idx]
                           for beam_idx in range(step_beam_size)]
            sampling_params = SamplingParams(
                max_tokens=3000, n=1, logprobs=0, stop="<end_of_reasoning>")
            # Record the number of input tokens
            for each_input in inputs_list:
                input_token_num_for_this_question += len(
                    tokenizer(each_input)['input_ids'])
            outputs = model.generate(inputs_list, sampling_params)
            total_rollout_times += step_beam_size

            candidates_list = []
            normalized_logp_list = []
            advantages_list = []
            output_text_list = []
            final_answer_info = []
            for jx in range(step_beam_size):
                output = outputs[jx].outputs[0]
                response = output.text.strip()
                candidates_list.append(response)
                normalized_logp_list.append(
                    output.cumulative_logprob / (len(output.token_ids)+1e-8))
                advantages_list.append(
                    output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx])
                all_cumulative_logprob_list[jx].append(
                    output.cumulative_logprob)
                token_num_list[jx].append(len(output.token_ids))
                output_text_list.append(response)
                all_token_num += len(output.token_ids)
                # Record information for traj info
                tem_answer_info = {'index': jx, 'text': response, 'adv': output.cumulative_logprob / (len(output.token_ids)+1e-8) - previous_q_value_list[jx],
                                   'logp': output.cumulative_logprob / (len(output.token_ids)+1e-8),
                                   'cumulative_logp': output.cumulative_logprob, 'token_num': len(output.token_ids)}
                final_answer_info.append(tem_answer_info)
                # Record the number of output tokens
                output_token_num_for_this_question += len(output.token_ids)
            traj_info['final_part']['final_answer'] = final_answer_info
            # At this point, only one index is selected
            if args.strategy == "cluster":
                # mask steps with adv less than 0
                tem_output_text_list = []
                tem_selected_index_list = []
                cur_cluster_info = {}
                if args.adv_above0 == "True":
                    mask = [adv > 0 for adv in advantages_list]
                    # ddi has no specific meaning, just represents the index
                    for ddi in range(len(mask)):
                        # Require adv greater than 0 and text not empty
                        if mask[ddi] and output_text_list[ddi] != '':
                            tem_selected_index_list.append(ddi)
                else:
                    # Remove empty strings
                    for dadagad in range(len(output_text_list)):
                        if output_text_list[dadagad] != '':
                            tem_selected_index_list.append(dadagad)

                if len(tem_selected_index_list) == 0:
                    # Use adv no replace
                    cur_cluster_info['state'] = 'fail'
                    # TODO: added a negative sign here
                    weights = softmax(
                        [-adv/temp for adv in advantages_list])
                    selected_index_final = np.random.choice(
                        len(weights), p=weights)
                else:
                    try:
                        tem_output_text_list = [
                            output_text_list[ddi] for ddi in tem_selected_index_list]
                        tem_advantages_list = [
                            advantages_list[ddi] for ddi in tem_selected_index_list]
                        vectorizer = TfidfVectorizer()
                        X = vectorizer.fit_transform(
                            tem_output_text_list)
                        k = 2  # TODO: may change later
                        kmeans = KMeans(n_clusters=k)
                        kmeans.fit(X)
                        cluster_labels = kmeans.labels_
                        cluster_list = [[] for _ in range(k)]

                        for aidx, cluster_label in enumerate(cluster_labels):
                            cluster_list[cluster_label].append(aidx)
                        cluster_list = [sorted(cluster)
                                        for cluster in cluster_list]
                        cluster0 = []
                        cluster1 = []
                        for aidx, cluster_label in enumerate(cluster_labels):
                            if cluster_label == 0:
                                cluster0.append(aidx)
                            else:
                                cluster1.append(aidx)
                        if len(cluster0) > len(cluster1):
                            cluster_adv_list = [
                                tem_advantages_list[ddi] for ddi in cluster0]
                            weights = softmax(
                                [adv/temp for adv in cluster_adv_list])
                            selected_index_in_cluster = np.random.choice(
                                len(weights), p=weights)
                            selected_index_in_tem = cluster0[selected_index_in_cluster]
                            selected_index_final = tem_selected_index_list[selected_index_in_tem]
                        else:
                            cluster_adv_list = [
                                tem_advantages_list[ddi] for ddi in cluster1]
                            weights = softmax(
                                [adv/temp for adv in cluster_adv_list])
                            selected_index_in_cluster = np.random.choice(
                                len(weights), p=weights)
                            selected_index_in_tem = cluster0[selected_index_in_cluster]
                            selected_index_final = tem_selected_index_list[selected_index_in_tem]
                        cur_cluster_info['state'] = 'success'
                        cur_cluster_info['cluster_result_cluster_idx'] = cluster_list
                        cur_cluster_info['cluster_result_answer_idx'] = [
                            [tem_selected_index_list[iodx] for iodx in cluster] for cluster in cluster_list]
                    except:
                        # Use adv no replace
                        cur_cluster_info['state'] = 'fail'
                        weights = softmax(
                            [adv/temp for adv in advantages_list])
                        selected_index_final = np.random.choice(
                            len(weights), p=weights)
                    cur_cluster_info['selected_idx_in_origin'] = selected_index_final
                traj_info['final_part']['cluster'] = cur_cluster_info

            ###################### generation final answer (end)######################

            whole_traj = previous_steps_list[selected_index_final] + "\n" + \
                candidates_list[selected_index_final]  # Final answer traj
            whole_traj_list = [previous_steps_list[beam_idx] + "\n" + candidates_list[beam_idx]
                               for beam_idx in range(step_beam_size)]  # Traj for each beam
            traj_info['token_num'] = all_token_num
            all_traj_info.append(traj_info)

            ###########################################################
            #           Write to result file  (all code below)        #
            ###########################################################
            result = {}
            result['id'] = i
            if "gsm" in args.data_path or "math" in args.data_path or "aime" in args.data_path:
                result['question'] = question
                result['ground_truth'] = test_data[i]['target']
            elif "reclor" in args.data_path or "logiqa" in args.data_path:
                result['question'] = 'Passage: ' + test_data[i]['context'] + '\nQuestion: ' + test_data[i]['question'] + \
                    f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
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
            print('output_token_num_for_this_question: ',
                  output_token_num_for_this_question)
            print('input_token_num_for_this_question: ',
                  input_token_num_for_this_question)
            print('all_output_token_num: ', all_output_token_num)
            print('all_input_token_num: ', all_input_token_num)
    end_time = time.time()
    time_span = end_time - start_time
    print(f"time: {time_span}")
    time_path = args.time_path + ffname + '.txt'
    with open(time_path, 'a') as f:
        f.write('time:  ' + str(time_span) + '\n')
        f.write('total:  ' + str(total_rollout_times) + '\n')
        f.write('save:  ' + str(saved_rollout_times) + '\n')
        f.write('num_rollout:  ' + str(num_rollout) + '\n')
        f.write('num_foresight:  ' + str(num_foresight) + '\n')
        f.write('step_beam_size:  ' + str(step_beam_size) + '\n')
        f.write('strategy:  ' + str(args.strategy) + '\n')
        f.write('final_select_strategy:  ' +
                str(args.final_select_strategy) + '\n')
        f.write('pruning_strategy:  ' + str(args.pruning_strategy) + '\n')
        f.write('depth_pruning_strategy:  ' +
                str(args.depth_pruning_strategy) + '\n')
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
