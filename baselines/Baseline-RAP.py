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
from typing import Generic, Optional, NamedTuple, Callable, Hashable, List, Tuple
import math

# TODO:把部分内容从main中放到这里了
parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default='gsm')
parser.add_argument('--model_id', type=str, default='llama3.1')
parser.add_argument('--data_path', type=str, default='/cpfs01/user/xufangzhi/o1/data/math_500_test.json')
parser.add_argument('--output_dir', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--step_beam_size', type=int, default=4)
parser.add_argument('--num_rollout', type=int, default=8) # num rollout其实就是每次expand的时候，llm生成的action的个数
parser.add_argument('--num_foresight', type=int, default=4)
parser.add_argument('--record_process', type=bool, default=True)
parser.add_argument('--strategy', type=str, default='tot_vote')
parser.add_argument('--time_path', type=str, default='/cpfs01/user/xufangzhi/o1/cluster_results/time/')
parser.add_argument('--file_name', type=str, default='RAP')
parser.add_argument('--n_iters', type=int, default=10) # 迭代次数 原本是10
parser.add_argument('--max_depth', type=int, default=5) # 最大深度 原本是10
parser.add_argument('--w_exp', type=float, default=1) # uct中的探索项
parser.add_argument('--uct_with_fast_reward', type=bool, default=True) # 在扩展select的时候，用RAP里面的uct式子。否则就从未访问的子节点中随机选一个
parser.add_argument('--simulate_choice', type=str, default='max') # 在simulate的时候，选择哪个子节点
parser.add_argument('--output_strategy', type=str, default='max_iter') # max_iter, follow_max最后用什么策略选择一条输出路径

args = parser.parse_args()
ffname = args.file_name
args.output_path = args.output_dir + ffname  + '.json'

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





class MCTSnode:
    def __init__(self, state:'Optional[List[str]]'=None, action=None, parent: 'Optional[MCTSnode]' = None, fast_reward:float = 0., is_terminal=False, question=None, depth=0):
        self.state = state # list of action text
        self.action = action
        self.parent = parent
        self.children : 'Optional[list[MCTSnode]]' = None
        self.fast_reward = fast_reward # 就是对于当前步骤yes or no的值，也可以用logp表示
        self.cum_rewards: list[float] = []
        self.is_terminal = is_terminal
        self.question = question
        self.depth = depth
        self.reward = 0.
    def Q(self):
        return sum(self.cum_rewards) / max(1, len(self.cum_rewards)) # back propagate的时候，会更新self.cum_rewards

def is_terminal_with_depth_limit(node: MCTSnode):
    return node.is_terminal or node.depth >= args.max_depth

def uct_select(node: MCTSnode):
    def _uct(x):
        return x.Q() + args.w_exp * np.sqrt(np.log(len(x.parent.cum_rewards)) / max(1, len(x.cum_rewards)))
    if args.uct_with_fast_reward or all(x.state is not None for x in node.children):
        return max(node.children, key=_uct)
    else:
        unvisited_children = filter(lambda x: x.state is None, node.children)
        return max(unvisited_children, key=lambda x: x.fast_reward)

def get_actions(state: "Optional[List[str]]", question=None):
    """根据当前的state，生成下一步的action, 并且返回每个action对应的normalized logp"""
    if "gsm" in args.data_path or "math" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'The question: ' + question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "reclor" in args.data_path or "logiqa" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question + '\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "strategy" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'The question is: ' + question + "\nAt the end, you must output 'Yes' or 'No' after 'The answer is: '." + '\nThe reasoning steps are:\n\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "cs" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'Passage: ' + question + '\n\nThe reasoning steps are:\n\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "gpqa" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'Passage: ' + question + '\n\nThe reasoning steps are:\n\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "arc" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'Passage: ' + question + '\n\nThe reasoning steps are:\n\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "scibench" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'The question: ' + question + '\n\nThe reasoning steps are:\n\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "truthfulqa_mc1" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': 'The question: ' + question + '\n\nThe reasoning steps are:\n\n'},
            {'role': 'assistant', 'content': ''}
        ]
    elif "humaneval" in args.data_path:
        chat = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': question + "\n"},
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
    
    reasoning_process = "The reasoning steps are:\n\n"
    for _ in range(len(state)):
        reasoning_process += state[_] + "\n"
    inputs_list = [inputs + reasoning_process] # 只有一个输入
    input_token_num = len(tokenizer(inputs_list[0])['input_ids'])
    # for each_input in inputs_list:
    #     input_token_num_for_this_question += len(tokenizer(each_input)['input_ids'])

    sampling_params = SamplingParams(max_tokens=1024, n=num_rollout, logprobs=0, temperature=0.6, stop=["\n", "<end_of_reasoning>"])
    # sampling_params = SamplingParams(max_tokens=1024 ,n=num_rollout, logprobs=0, best_of=4, temperature=0, use_beam_search=True, stop=["\n", "<end_of_reasoning>"])
    # 根据当前的状态，foresight
    outputs = model.generate(inputs_list, sampling_params)

    output_token_num = 0
    action_list = []
    normalized_logp_list = []
    for idx in range(num_rollout):
        output = outputs[0].outputs[idx]
        response = output.text.strip()
        action_list.append(response)
        output_token_num += len(output.token_ids)
        normalized_logp_list.append(output.cumulative_logprob / (len(output.token_ids)+1e-8))
        # tem_normalized_logp = output.cumulative_logprob / (len(output.token_ids)+1e-8)
        # normalized_p_list.append(np.exp(tem_normalized_logp))
    return action_list, normalized_logp_list, input_token_num, output_token_num

def simulate_choice(fast_rewards):
    if args.simulate_choice == 'max':
        return np.argmax(fast_rewards)

def select(node: MCTSnode):
    path = []
    while True:
        path.append(node)
        if node.is_terminal or node.children is None or node.depth >= args.max_depth:
            return path
        node = uct_select(node)

def expand(node: MCTSnode, question=None):
    """让llm基于当前状态，生成n_action个子节点，（刚生成的时候，每个子节点的state是none)"""
    if node.state is None:
        node.state = node.parent.state + [node.action]
        # reward is calculated after the state is updated, so that the
        # information can be cached and passed from the world model
        # to the reward function with **aux without repetitive computation
        node.reward = node.fast_reward
        def is_terminal(state):
            tem_str = state[-1]
            if "the answer is" not in tem_str.lower().strip(): # 没产生最终答案
                return False
            if "the answer is" == tem_str.lower().strip(): # 只生成了这一句话
                return False
            return True
        node.is_terminal = is_terminal(node.state)

    if node.is_terminal:
        return 0, 0

    children = []
    actions, normalized_logp, input_token_num, output_token_num = get_actions(node.state, question=question) # 根据当前状态，让LLM生成几个可能的action
    # 简单使用logp来做reward
    for idx, action in enumerate(actions):
        fast_reward = normalized_logp[idx]
        child = MCTSnode(state=None, action=action, parent=node,
                            fast_reward=fast_reward, depth=node.depth+1, question=question)
        children.append(child)

    node.children = children
    return input_token_num, output_token_num

def simulate(path, question=None):
    node = path[-1]
    input_token_num, output_token_num = 0, 0
    while True:
        if node.state is None:
            tem_input_token_num, tem_output_token_num = expand(node, question=question)
            input_token_num += tem_input_token_num
            output_token_num += tem_output_token_num
        if is_terminal_with_depth_limit(node) or len(node.children) == 0:
            return input_token_num, output_token_num
        fast_rewards = [child.fast_reward for child in node.children]
        node = node.children[simulate_choice(fast_rewards)]
        path.append(node)

def back_propagate(path):
    rewards = []
    cum_reward = -math.inf
    for node in reversed(path):
        rewards.append(node.reward)
        cum_reward = sum(rewards[::-1])
        node.cum_rewards.append(cum_reward)
    return cum_reward

if __name__ == '__main__':
    iadx = 0
    all_output_token_num = 0
    all_input_token_num = 0
    with open(OUTPUT_PATH, "w") as f:
        for i in range(len(test_data)):
            output_token_num_for_this_question = 0
            input_token_num_for_this_question = 0
            problem_start_time = time.time()

            if "reclor" in args.data_path or "logiqa" in args.data_path:
                question = 'Passage: ' + test_data[i]['context'] + '\nQuestion: '+ test_data[i]['question'] + f"\nA. {test_data[i]['answers'][0]}\nB. {test_data[i]['answers'][1]}\nC. {test_data[i]['answers'][2]}\nD. {test_data[i]['answers'][3]}"
            elif "humaneval" in args.data_path: 
                question = test_data[i]['prompt']
            else:
                question = test_data[i]['input']

            root = MCTSnode(state=['root'])
            _output_cum_reward = -math.inf
            _output_iter = []
            # =================== start of iterate ============================ #
            for _ in range(args.n_iters):
                path = select(root)
                if not is_terminal_with_depth_limit(path[-1]):
                    tem_input_token_num, tem_output_token_num = expand(path[-1], question=question)
                    input_token_num_for_this_question += tem_input_token_num
                    output_token_num_for_this_question += tem_output_token_num
                    tem_input_token_num, tem_output_token_num = simulate(path, question=question) # 这里好像只simulate了一次
                    input_token_num_for_this_question += tem_input_token_num
                    output_token_num_for_this_question += tem_output_token_num
                cum_reward = back_propagate(path)
                if args.output_strategy == 'max_iter' and path[-1].is_terminal and cum_reward > _output_cum_reward:
                    _output_cum_reward = cum_reward
                    _output_iter = path
                if args.output_strategy == 'last_iter': # 不管这个path是否达到终点
                    _output_cum_reward = cum_reward
                    _output_iter = path
                if args.output_strategy == 'last_terminal_iter' and path[-1].is_terminal:
                    _output_cum_reward = cum_reward
                    _output_iter = path
                final_path = path
            # =================== end of iterate ============================ #

            if args.output_strategy == 'follow_max': # 从root开始，一直往下找reward最大的,reward其实就是fast_reward，也就是
                _output_iter = []
                cur = root
                while True:
                    _output_iter.append(cur)
                    if cur.is_terminal:
                        break
                    visited_children = [x for x in cur.children if x.state is not None]
                    if len(visited_children) == 0:
                        break
                    cur = max(visited_children, key=lambda x: x.reward)
                # _output_cum_reward = self.cum_reward([node.reward for node in self._output_iter[1::-1]])

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


            result['response'] = "The reasoning steps are:\n\n" + ''.join([x.action for x in _output_iter[1:]]) # 把根节点排除掉，因为就没用
            result['response_all_beams'] = [result['response']] # TODO:后面可以更改，让每个问题产生多个response，把结果存到这里
            probelm_stop_time = time.time()
            print(f"problem {i} time usage: {probelm_stop_time - problem_start_time}")
            all_output_token_num += output_token_num_for_this_question
            all_input_token_num += input_token_num_for_this_question
            print(f"output_token_num: {output_token_num_for_this_question}")
            print(f"input_token_num: {input_token_num_for_this_question}")
            print(f"all_output_token_num: {all_output_token_num}")
            print(f"all_input_token_num: {all_input_token_num}")
            f.write(json.dumps(result) + '\n')
            f.flush() 
    end_time = time.time()
    time_span = end_time - start_time
    print(f"time: {time_span}")
    time_path = args.time_path + ffname + '.txt'
    with open(time_path, 'a') as f:
        f.write('time:  ' + str(time_span) + '\n')
        # f.write('total:  ' + str(total_rollout_times) + '\n')
        # f.write('save:  ' + str(saved_rollout_times) + '\n')
        f.write('num_rollout:  ' + str(num_rollout) + '\n')
        f.write('num_foresight:  ' + str(num_foresight) + '\n')
        f.write('all_output_token_num:  ' + str(all_output_token_num) + '\n')
        f.write('all_input_token_num:  ' + str(all_input_token_num) + '\n')
        f.write('n_iters:  ' + str(args.n_iters) + '\n')
        f.write('max_depth:  ' + str(args.max_depth) + '\n')
        f.write('w_exp:  ' + str(args.w_exp) + '\n')
        f.write('uct_with_fast_reward:  ' + str(args.uct_with_fast_reward) + '\n')
        f.write('simulate_choice:  ' + str(args.simulate_choice) + '\n')
        f.write('output_strategy:  ' + str(args.output_strategy) + '\n')
    # print('total: ', total_rollout_times)
    # print('save: ', saved_rollout_times)
