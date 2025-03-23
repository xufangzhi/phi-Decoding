# -*- coding: utf-8 -*-
# Phi-Decoding: A decoding algorithm that combines clustering and sampling strategies
# This implementation uses TF-IDF vectorization and K-means clustering for response selection
# Warning: This implementation may be unstable and requires further testing
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import json
import os
import argparse
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
from data.math_example import (
    MATH_POT_FEW_SHOT,
    MATH_COT_FEW_SHOT,
    GSM_COT_8_SHOT,
    MATH_COT_4_SHOT
)

# set visible gpus
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


# Constants for algorithm configuration
INF = 10  # Used for initialization of min/max values
TEMPERATURE = 0.1  # Temperature parameter for softmax sampling


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Phi-Decoding Algorithm")

    # Model configuration
    parser.add_argument('--model_id', type=str, default='llama3.1',
                        help='Model identifier')
    parser.add_argument('--model_path', type=str, default='/data/xrm/yh/test_unsloth/Meta-Llama-3.1-8B-Instruct',
                        help='Model path')
    parser.add_argument('--gpus', type=int, default=2,
                        help='Number of GPUs to use')

    # Data configuration
    parser.add_argument('--datasets', type=str, default='gsm',
                        help='Dataset type')  # gsm, math, reclor, logiqa, gpqa, arc
    parser.add_argument('--data_path', type=str,
                        default='./data/gsm_test.json',
                        help='Path to input data')
    parser.add_argument('--output_dir', type=str,
                        default='./results/',
                        help='Output directory for results')

    # Algorithm parameters
    parser.add_argument('--step_beam_size', type=int, default=4,
                        help='Beam size for each step')
    parser.add_argument('--num_rollout', type=int, default=4,
                        help='Number of rollouts')
    parser.add_argument('--num_foresight', type=int, default=8,
                        help='Number of foresight steps')
    parser.add_argument('--strategy', type=str, default='cluster',
                        help='Response selection strategy')
    parser.add_argument('--width_pruning_strategy', type=str, default='low_sigma',
                        help='Width pruning strategy')
    parser.add_argument('--depth_pruning_strategy', type=str, default='cluster',
                        help='Depth pruning strategy')
    parser.add_argument('--cluster_num', type=int, default=3,
                        help='Number of clusters for clustering strategy')
    parser.add_argument('--threshold', type=float, default=0.69,
                        help='Threshold for early stopping')
    parser.add_argument('--least_foresight_num', type=int, default=4,
                        help='Minimum number of foresight steps')
    parser.add_argument('--sigma_rate', type=float, default=1.0,
                        help='Sigma rate for width pruning')

    # Execution configuration
    parser.add_argument('--record_process', type=bool, default=True,
                        help='Whether to record the decoding process')
    parser.add_argument('--file_name', type=str, default='test_3',
                        help='Output file name')
    parser.add_argument('--time_path', type=str,
                        default='./results/time/',
                        help='Path to save timing information')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    return parser.parse_args()


def softmax(x):
    """
    Compute softmax values for the input array
    Args:
        x: Input array of values
    Returns:
        Softmax probabilities
    """
    e_x = np.exp(np.array(x))
    return e_x / e_x.sum(axis=0)


class PhiDecoder:
    """
    Main class for phi-decoding algorithm implementation.
    Combines clustering and sampling strategies for response selection.
    """

    def __init__(self, args):
        """
        Initialize the decoder
        Args:
            args: Command line arguments containing configuration
        """
        self.args = args
        self.model = None
        self.tokenizer = None
        self.initialize_model()

    def initialize_model(self):
        """Initialize the language model and tokenizer"""
        model_path = self._get_model_path()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, max_length=32768)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LLM(
            model=model_path,
            tensor_parallel_size=self.args.gpus,
            trust_remote_code=True,
            max_model_len=32768
        )

        np.random.seed(self.args.seed)

    def _get_model_path(self):
        """Get the appropriate model path"""
        return self.args.model_path

    def get_system_prompt(self, dataset_type):
        """
        Get the appropriate system prompt based on dataset type
        Args:
            dataset_type: Type of dataset (e.g., 'gsm', 'math', etc.)
        Returns:
            System prompt string
        """
        prompts = {
            "gsm": f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{GSM_COT_8_SHOT}",
            "math": f"Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{MATH_COT_4_SHOT}",
            "reclor": f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Please finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{LOGIC_MRC_COT_4_SHOT}\n",
            "logiqa": f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D'. Read the question and options thoroughly and select the correct answer from the four answer labels. Please finish the reasoning with <end_of_reasoning>.\nI will give you some examples for reference.\n{LOGIC_MRC_COT_4_SHOT}\n",
            "gpqa": f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response.",
            "arc": f"Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response.",
        }

        # for R1, we use the following system prompt
        if hasattr(self.args, 'model_id') and 'r1' in self.args.model_id:
            return "A conversation between user and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."

        return prompts.get(dataset_type, "")

    def cluster_responses(self, responses, advantages):
        """
        Cluster responses using TF-IDF and K-means
        Args:
            responses: List of response texts
            advantages: List of advantage values for each response
        Returns:
            Tuple of (clusters, cluster_info)
        """
        # Filter out empty responses
        valid_indices = [i for i, r in enumerate(responses) if r.strip()]
        if len(valid_indices) < self.args.step_beam_size:
            return None, {"state": "cannot cluster"}

        try:
            valid_responses = [responses[i] for i in valid_indices]

            # Vectorize responses
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(valid_responses)

            # Perform clustering
            kmeans = KMeans(n_clusters=2)  # Using 2 clusters as per paper
            kmeans.fit(X)

            # Group responses by cluster
            clusters = [[] for _ in range(2)]
            for idx, label in enumerate(kmeans.labels_):
                clusters[label].append(valid_indices[idx])

            return clusters, {
                "state": "success",
                "cluster_sizes": [len(c) for c in clusters]
            }

        except Exception as e:
            return None, {"state": "fail", "error": str(e)}

    def select_response(self, responses, logprobs, advantages):
        """Select final response based on strategy"""
        if self.args.strategy == "cluster":
            # filter out empty responses
            valid_indices = []
            for idx, response in enumerate(responses):
                if response.strip() != '':
                    valid_indices.append(idx)

            if len(valid_indices) == 0:
                print(
                    'all responses in the final generation are empty, use -adv no replace')
                # beacuse current responses are empty, add '-' to maximize the advantage
                weights = softmax([-adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            if len(valid_indices) < self.args.step_beam_size:
                # if the number of valid responses is less than step_beam_size, use adv no replace
                print('valid responses are less than step_beam_size, use adv no replace')
                weights = softmax([adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

            try:
                # prepare cluster data
                valid_responses = [responses[i] for i in valid_indices]
                valid_advantages = [advantages[i] for i in valid_indices]

                # execute TF-IDF vectorization and clustering
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(valid_responses)
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(X)
                cluster_labels = kmeans.labels_

                # build cluster list
                cluster_list = [[] for _ in range(2)]
                for idx, label in enumerate(cluster_labels):
                    cluster_list[label].append(idx)
                cluster_list = [sorted(cluster) for cluster in cluster_list]

                cluster0 = cluster_list[0]
                cluster1 = cluster_list[1]
                if len(cluster0) > len(cluster1):
                    cluster_adv_list = [valid_advantages[ddi]
                                        for ddi in cluster0]
                    weights = softmax(
                        [adv/TEMPERATURE for adv in cluster_adv_list])
                    selected_index_in_cluster = np.random.choice(
                        len(weights), p=weights)
                    selected_index_in_tem = cluster0[selected_index_in_cluster]
                    selected_index_final = valid_indices[selected_index_in_tem]
                    return selected_index_final
                else:
                    cluster_adv_list = [valid_advantages[ddi]
                                        for ddi in cluster1]
                    weights = softmax(
                        [adv/TEMPERATURE for adv in cluster_adv_list])
                    selected_index_in_cluster = np.random.choice(
                        len(weights), p=weights)
                    selected_index_in_tem = cluster1[selected_index_in_cluster]
                    selected_index_final = valid_indices[selected_index_in_tem]
                    return selected_index_final

            except Exception as e:
                print('cannot select response based on cluster, use adv no replace')
                weights = softmax([adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights)

        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def process_example(self, example, system_prompt):
        """
        Process a single example through the phi-decoding pipeline
        Args:
            example: Input example containing question and other fields
            system_prompt: System prompt for the model
        Returns:
            Dictionary containing results and statistics
        """
        # Initialize tracking variables
        token_stats = {"input": 0, "output": 0}
        rollout_stats = {"total": 0, "saved": 0}

        # Initialize trajectory pools
        traj_pool = [[] for _ in range(self.args.num_foresight)]
        step_pool = [[] for _ in range(self.args.num_foresight)]
        prob_pool = [[] for _ in range(self.args.num_foresight + 1)]
        adv_pool = [[] for _ in range(self.args.num_foresight + 1)]

        # Initialize beam states
        previous_steps = ["The reasoning steps are:\n\n" for _ in range(
            self.args.step_beam_size)]
        previous_values = [0.0 for _ in range(self.args.step_beam_size)]

        # Initialize trajectory information
        if "reclor" in self.args.datasets or "logiqa" in self.args.datasets:
            traj_info = {
                'question_idx': example.get('id', 0),
                'question': f"Passage: {example['context']}\nQuestion: {example['question']}\n" +
                f"A. {example['answers'][0]}\nB. {example['answers'][1]}\n" +
                f"C. {example['answers'][2]}\nD. {example['answers'][3]}",
                'ground_truth': example.get('label'),
                'foresight_part': [],  # Will be filled during each step
                'final_part': {},      # Will be filled during final generation
                'config': {            # Add configuration information
                    'num_rollout': self.args.num_rollout,
                    'num_foresight': self.args.num_foresight,
                    'step_beam_size': self.args.step_beam_size,
                    'strategy': self.args.strategy,
                    'width_pruning_strategy': self.args.width_pruning_strategy,
                    'depth_pruning_strategy': self.args.depth_pruning_strategy,
                    'threshold': self.args.threshold,
                    'sigma_rate': self.args.sigma_rate,
                    'cluster_num': self.args.cluster_num
                }
            }
        else:
            traj_info = {
                'question_idx': example.get('id', 0),
                'question': example['input'],
                'ground_truth': example.get('target'),
                'foresight_part': [],  # Will be filled during each step
                'final_part': {},      # Will be filled during final generation
                'config': {            # Add configuration information
                    'num_rollout': self.args.num_rollout,
                    'num_foresight': self.args.num_foresight,
                    'step_beam_size': self.args.step_beam_size,
                    'strategy': self.args.strategy,
                    'width_pruning_strategy': self.args.width_pruning_strategy,
                    'depth_pruning_strategy': self.args.depth_pruning_strategy,
                    'threshold': self.args.threshold,
                    'sigma_rate': self.args.sigma_rate,
                    'cluster_num': self.args.cluster_num
                }
            }

        # Multi-step reasoning
        for step in range(self.args.num_foresight):
            step_results = self._process_step(
                example,
                system_prompt,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats,
                traj_info  # Pass trajectory information
            )

            # Check early stopping condition
            if self._should_stop_early(step_results, step):
                break

            # Update state for next step
            previous_steps = step_results["next_steps"]
            previous_values = step_results["next_values"]

            # Record step results
            traj_pool[step] = step_results["trajectories"]
            step_pool[step] = step_results["steps"]
            prob_pool[step] = step_results["logprobs"]
            adv_pool[step] = step_results["advantages"]

        # Generate final response
        final_result = self._generate_final_response(
            example,
            system_prompt,
            previous_steps,
            previous_values,
            token_stats,
            rollout_stats,
            traj_info  # Pass trajectory information
        )

        # Record token statistics
        traj_info['token_num'] = token_stats["input"] + token_stats["output"]

        return {
            # "response_in_the_final_generation": final_result["response_in_the_final_generation"],
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"]
            },
            "traj_info": traj_info  # Add trajectory information to return result
        }

    def _process_step(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """Process a single reasoning step"""
        stop_foresight = False
        # first stage: generate incomplete responses
        all_inputs = []
        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template_for_first_stage(
                example, system_prompt)
            if self.args.model_id == "mistral":
                chat[1]['content'] = system_prompt + "\n" + chat[1]['content']
                chat = chat[1:]
            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()
            inputs = inputs + previous_steps[beam_idx]
            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])
            all_inputs.append(inputs)

        sampling_params = SamplingParams(
            max_tokens=1024, n=self.args.num_rollout, logprobs=0, temperature=0.6, stop=["\n", "<end_of_reasoning>"])

        outputs = self.model.generate(
            all_inputs,
            sampling_params
        )

        rollout_stats["total"] += self.args.num_rollout * \
            self.args.step_beam_size

        # collect the results of the first stage
        all_responses_first_stage = []
        all_logprobs_first_stage = []
        all_advantages_first_stage = []
        all_token_nums_first_stage = []

        for beam_idx, beam_outputs in enumerate(outputs):
            for output in beam_outputs.outputs:
                response = output.text.strip()
                logprob = output.cumulative_logprob / \
                    (len(output.token_ids) + 1e-8)
                advantage = logprob - previous_values[beam_idx]

                all_responses_first_stage.append(response)
                all_logprobs_first_stage.append(logprob)
                all_advantages_first_stage.append(advantage)
                all_token_nums_first_stage.append(len(output.token_ids))
                token_stats["output"] += len(output.token_ids)

        # prune the responses based on width pruning_strategy
        if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
            keep_foresight_list = []
            if self.args.width_pruning_strategy == "low_sigma":
                # calculate the mean and standard deviation of logprobs
                mean = np.mean(all_logprobs_first_stage)
                std = np.std(all_logprobs_first_stage)

                # keep the samples with logprob higher than mean - sigma_rate * std
                for idx, logp in enumerate(all_logprobs_first_stage):
                    if logp > mean - self.args.sigma_rate * std:
                        keep_foresight_list.append(idx)

            # if the number of kept samples is less than step_beam_size, then supplement
            if len(keep_foresight_list) < self.args.step_beam_size:
                weights = softmax(
                    [logp/TEMPERATURE for logp in all_logprobs_first_stage])
                num_to_add = self.args.step_beam_size - \
                    len(keep_foresight_list)
                available_indices = [i for i in range(
                    len(all_logprobs_first_stage)) if i not in keep_foresight_list]
                if available_indices:
                    available_weights = [weights[i] for i in available_indices]
                    available_weights = [w/sum(available_weights)
                                         for w in available_weights]
                    additional_indices = np.random.choice(
                        available_indices,
                        # size=min(num_to_add, len(available_indices)),
                        size=num_to_add,
                        p=available_weights,
                        replace=False
                    ).tolist()
                    keep_foresight_list.extend(additional_indices)

            keep_foresight_list.sort()

            # update the statistics
            rollout_stats["saved"] += (self.args.step_beam_size *
                                       self.args.num_rollout - len(keep_foresight_list))

            # only keep the selected samples
            filtered_responses = [all_responses_first_stage[i]
                                  for i in keep_foresight_list]
            filtered_logprobs = [all_logprobs_first_stage[i]
                                 for i in keep_foresight_list]
            filtered_advantages = [all_advantages_first_stage[i]
                                   for i in keep_foresight_list]
            filtered_beam_indices = [
                i // self.args.num_rollout for i in keep_foresight_list]

            all_responses = filtered_responses

        # second stage: complete the responses
        completion_inputs = []
        for idx in range(len(keep_foresight_list)):
            response = all_responses[idx]
            # if response.strip() != '':
            chat = self._prepare_chat_template(example, system_prompt)
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            chat[-1]["content"] = previous_steps[beam_idx] + response

            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            completion_inputs.append(inputs)
            token_stats["input"] += len(self.tokenizer(inputs)
                                        ["input_ids"])

        # generate the completed responses
        sampling_params = SamplingParams(
            max_tokens=1024,
            n=1,
            logprobs=0,
            stop=["<end_of_reasoning>"]
        )

        completion_outputs = self.model.generate(
            completion_inputs, sampling_params)
        rollout_stats["total"] += len(completion_inputs)

        # collect the results of the second stage
        completed_responses = []
        completed_logprobs = []
        completed_advantages = []

        for idx, outputs in enumerate(completion_outputs):
            output = outputs.outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / \
                (len(output.token_ids) + 1e-8)
            beam_idx = keep_foresight_list[idx] // self.args.num_rollout
            advantage = logprob - previous_values[beam_idx]

            completed_responses.append(response)
            completed_logprobs.append(logprob)
            completed_advantages.append(advantage)
            token_stats["output"] += len(output.token_ids)

        # third stage: cluster and select the completed responses
        try:
            # execute TF-IDF vectorization and clustering
            vectorizer = TfidfVectorizer()
            X = vectorizer.fit_transform(completed_responses)
            kmeans = KMeans(n_clusters=self.args.cluster_num)
            kmeans.fit(X)
            cluster_labels = kmeans.labels_

            # build the cluster list
            cluster_list = [[] for _ in range(self.args.cluster_num)]
            for idx, label in enumerate(cluster_labels):
                cluster_list[label].append(idx)
            cluster_list = [sorted(cluster) for cluster in cluster_list]

            # calculate the cluster weights and advantage weights
            cluster_len_ratio = [len(cluster)/len(completed_responses)
                                 for cluster in cluster_list]
            per_sample_cluster_len_ratio = [
                cluster_len_ratio[cluster_labels[i]] for i in range(len(completed_responses))]
            cluster_weights = softmax(per_sample_cluster_len_ratio)
            adv_weights = softmax(
                [adv/TEMPERATURE for adv in completed_advantages])

            # combine the weights
            weights = [(cluster_weights[ii] + adv_weights[ii]) /
                       2 for ii in range(len(completed_responses))]

            # select the samples
            selected = np.random.choice(
                len(weights),
                size=self.args.step_beam_size,
                p=weights,
                replace=False
            ).tolist()

            sizes = np.bincount(cluster_labels)
            largest_ratio = max(sizes) / len(completed_responses)

            if largest_ratio >= self.args.threshold:
                stop_foresight = True

            # Record information after generating first stage responses
            step_info = {
                'first_stage': {
                    'responses': all_responses_first_stage,
                    'logprobs': all_logprobs_first_stage,
                    'advantages': all_advantages_first_stage,
                    'token_nums': all_token_nums_first_stage
                }
            }

            # Record information after width pruning
            if self.args.width_pruning_strategy != "none" and self.args.width_pruning_strategy != "":
                step_info['width_pruning'] = {
                    'keep_indices': keep_foresight_list,
                    'filtered_responses': filtered_responses,
                    'filtered_logprobs': filtered_logprobs,
                    'filtered_advantages': filtered_advantages,
                    'filtered_beam_indices': filtered_beam_indices
                }

            # Record information after second stage completion
            step_info['second_stage'] = {
                'responses': completed_responses,
                'logprobs': completed_logprobs,
                'advantages': completed_advantages
            }

            # Record information after clustering and selection
            step_info['clustering'] = {
                'cluster_labels': cluster_labels.tolist(),
                'cluster_sizes': [len(cluster) for cluster in cluster_list],
                'cluster_weights': cluster_weights.tolist(),
                'adv_weights': adv_weights.tolist(),
                'combined_weights': weights,
                'selected_indices': selected
            }

            # Record final selection results
            step_info['final'] = {
                'selected_steps': [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + all_responses_first_stage[keep_foresight_list[idx]] + "\n" for idx in selected],
                'selected_values': [completed_logprobs[idx] for idx in selected],
                'selected_indices': selected
            }

            # Add current step information to trajectory information
            traj_info['foresight_part'].append(step_info)

            return {
                "next_steps": [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + all_responses_first_stage[keep_foresight_list[idx]] + "\n" for idx in selected],
                "next_values": [completed_logprobs[idx] for idx in selected],
                "trajectories": completed_responses,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": completed_logprobs,
                "advantages": completed_advantages,
                "stop_foresight": stop_foresight
            }

        except Exception as e:
            print(
                'when cluster during intermediate steps, error occurs, use adv no replace')
            weights = softmax(
                [adv/TEMPERATURE for adv in completed_advantages])
            selected = np.random.choice(
                len(weights),
                size=self.args.step_beam_size,
                p=weights,
                replace=False
            ).tolist()

            return {
                "next_steps": [previous_steps[keep_foresight_list[idx]//self.args.num_rollout] + all_responses_first_stage[keep_foresight_list[idx]] + "\n" for idx in selected],
                "next_values": [all_logprobs_first_stage[idx] for idx in selected],
                "trajectories": all_responses_first_stage,
                "steps": [keep_foresight_list[idx] for idx in selected],
                "logprobs": all_logprobs_first_stage,
                "advantages": all_advantages_first_stage,
                "stop_foresight": stop_foresight
            }

    def _should_stop_early(self, step_results, current_step):
        """Check if early stopping conditions are met"""
        if current_step < self.args.least_foresight_num:
            return False

        just_stop = True
        first_response = step_results["trajectories"][0]
        for response in step_results["trajectories"][1:]:
            if response != first_response:
                just_stop = False
                break

        if just_stop:
            print(
                f'Early stopping at depth {current_step} (all responses are the same)')
            return True

        if self.args.depth_pruning_strategy == "cluster":
            # Check if responses are becoming similar
            if step_results["stop_foresight"]:
                print(
                    f'Early stopping at depth {current_step} (max cluster ratio >= args.threshold)')
                return True

        return False

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats, traj_info):
        """Generate final response after multi-step reasoning"""
        # Prepare input for each beam
        all_inputs = []
        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template(example, system_prompt)
            chat[-1]["content"] = previous_steps[beam_idx]

            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])
            all_inputs.append(inputs)

        # parallel generate all beam responses
        sampling_params = SamplingParams(
            max_tokens=3000,
            n=1,
            logprobs=0,
            stop=["<end_of_reasoning>"]
        )
        outputs = self.model.generate(all_inputs, sampling_params)

        rollout_stats["total"] += self.args.step_beam_size

        # Collect all response results
        all_responses = []
        all_logprobs = []
        all_advantages = []
        all_combined_responses = []

        for beam_idx, beam_outputs in enumerate(outputs):
            output = beam_outputs.outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / \
                (len(output.token_ids) + 1e-8)
            advantage = logprob - previous_values[beam_idx]

            # Combine previous_steps and new response
            combined_response = previous_steps[beam_idx] + response
            all_combined_responses.append(combined_response)
            all_responses.append(response)
            all_logprobs.append(logprob)
            all_advantages.append(advantage)
            token_stats["output"] += len(output.token_ids)

        # Select final response
        selected_idx = self.select_response(
            all_responses,
            all_logprobs,
            all_advantages
        )

        # Record final results
        traj_info['final_part']['responses'] = all_combined_responses
        traj_info['final_part']['responses_in_the_final_generation'] = all_responses
        traj_info['final_part']['logprobs'] = all_logprobs
        traj_info['final_part']['advantages'] = all_advantages
        traj_info['final_part']['selected_idx'] = selected_idx

        return {
            "response": previous_steps[selected_idx] + all_responses[selected_idx],
            # "response_in_the_final_generation": all_responses[selected_idx],
            "trajectories": {
                "responses": all_responses,
                "logprobs": all_logprobs,
                "advantages": all_advantages,
                "selected_idx": selected_idx
            }
        }

    def _prepare_chat_template(self, example, system_prompt):
        """
        Prepare chat template based on dataset type
        Args:
            example: Input example
            system_prompt: System prompt
        Returns:
            List of chat messages
        """
        if "gsm" in self.args.datasets or "math" in self.args.datasets:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user",
                    "content": f"The question: {example['input']}\nPlease directly output the reasoning steps.\n"},
                {"role": "assistant", "content": ""}
            ]
        elif "reclor" in self.args.datasets or "logiqa" in self.args.datasets:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Passage: {example['context']}\nQuestion: {example['question']}\n" +
                    f"A. {example['answers'][0]}\nB. {example['answers'][1]}\nC. {example['answers'][2]}\nD. {example['answers'][3]}\n"},
                {"role": "assistant", "content": ""}
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": ""}
            ]

    def _prepare_chat_template_for_first_stage(self, example, system_prompt):
        """
        Prepare chat template based on dataset type
        Args:
            example: Input example
            system_prompt: System prompt
        Returns:
            List of chat messages
        """
        if "gsm" in self.args.datasets or "math" in self.args.datasets:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user",
                    "content": f"The question: {example['input']}\nPlease directly follow the previous reasoning steps (if provided) and generate the remaining ones.\n"},
                {"role": "assistant", "content": ""}
            ]
        elif "reclor" in self.args.datasets or "logiqa" in self.args.datasets:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Passage: {example['context']}\nQuestion: {example['question']}\n" +
                    f"A. {example['answers'][0]}\nB. {example['answers'][1]}\nC. {example['answers'][2]}\nD. {example['answers'][3]}\n"},
                {"role": "assistant", "content": ""}
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["input"]},
                {"role": "assistant", "content": ""}
            ]


def main():
    """Main execution function"""
    args = parse_arguments()
    decoder = PhiDecoder(args)

    with open(args.data_path) as f:
        test_data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.time_path, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")

    # Record start time
    start_time = time.time()

    # Statistics
    total_stats = {
        "total_rollouts": 0,
        "saved_rollouts": 0,
        "input_tokens": 0,
        "output_tokens": 0
    }

    # Used to store all trajectory information
    all_traj_info = []

    # Process each test example
    for i, example in enumerate(test_data):
        # try:
        # Generate system prompt
        system_prompt = decoder.get_system_prompt(args.datasets)

        # Process example
        result = decoder.process_example(example, system_prompt)

        # Update statistics
        total_stats["total_rollouts"] += result["rollout_stats"]["total"]
        total_stats["saved_rollouts"] += result["rollout_stats"]["saved"]
        total_stats["input_tokens"] += result["token_stats"]["input"]
        total_stats["output_tokens"] += result["token_stats"]["output"]

        # Add trajectory information
        result["traj_info"]["question_idx"] = i
        all_traj_info.append(result["traj_info"])

        # Prepare output result
        output_result = {
            "id": i,
            "question": example["input"],
            "ground_truth": example.get("target"),
            "response": result["response"]
        }

        # Write result to main output file
        with open(output_path, "a") as f:
            f.write(json.dumps(output_result) + "\n")

        print(
            f'output_token_num_for_question{i}: {result["token_stats"]["output"]}')
        print(
            f'input_token_num_for_question{i}: {result["token_stats"]["input"]}')
        print(f'all_output_token_num: {total_stats["output_tokens"]}')
        print(f'all_input_token_num: {total_stats["input_tokens"]}')

        # Save trajectory information
        if args.record_process:
            traj_path = os.path.join(
                args.time_path, f"TRAJ_INFO-{args.file_name}.json")
            with open(traj_path, "w") as f:
                json.dump(all_traj_info, f, indent=2)

    # Calculate total time
    end_time = time.time()
    time_span = end_time - start_time

    # Save time information to separate file
    time_info_path = os.path.join(args.time_path, f"{args.file_name}.txt")
    with open(time_info_path, "w") as f:
        f.write(f'time:  {time_span}\n')
        f.write(f'total:  {total_stats["total_rollouts"]}\n')
        f.write(f'save:  {total_stats["saved_rollouts"]}\n')
        f.write(f'num_rollout:  {args.num_rollout}\n')
        f.write(f'num_foresight:  {args.num_foresight}\n')
        f.write(f'step_beam_size:  {args.step_beam_size}\n')
        f.write(f'strategy:  {args.strategy}\n')
        f.write(f'width_pruning_strategy:  {args.width_pruning_strategy}\n')
        f.write(f'depth_pruning_strategy:  {args.depth_pruning_strategy}\n')
        f.write(f'threshold:  {args.threshold}\n')
        f.write(f'sigma_rate:  {args.sigma_rate}\n')
        f.write(f'cluster_num:  {args.cluster_num}\n')
        f.write(f'all_input_token_num:  {total_stats["input_tokens"]}\n')
        f.write(f'all_output_token_num:  {total_stats["output_tokens"]}\n')

    print('total rollouts: ', total_stats["total_rollouts"])
    print('saved rollouts: ', total_stats["saved_rollouts"])
    print('all_output_token_num: ', total_stats["output_tokens"])
    print('all_input_token_num: ', total_stats["input_tokens"])


if __name__ == "__main__":
    main()
