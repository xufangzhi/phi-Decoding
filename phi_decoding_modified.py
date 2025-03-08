# Phi-Decoding: A decoding algorithm that combines clustering and sampling strategies
# This implementation uses TF-IDF vectorization and K-means clustering for response selection

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

# Constants for algorithm configuration
INF = 10  # Used for initialization of min/max values
TEMPERATURE = 0.1  # Temperature parameter for softmax sampling


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
            model_path, max_length=32768, trust_remote_code=True)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = LLM(
            model=model_path,
            tensor_parallel_size=self.args.gpus,
            trust_remote_code=True
        )

    def _get_model_path(self):
        """Get the appropriate model path based on model ID"""
        model_paths = {
            "llama3.1": "/nas/shared/NLP_A100/hf_hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
            "qwen7B": "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75",
            "qwen3B": "/nas/shared/NLP_A100/hf_hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
            "mistral": "/nas/shared/NLP_A100/hf_hub/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/83e9aa141f2e28c82232fea5325f54edf17c43de",
            "gemma": "/nas/shared/NLP_A100/hf_hub/models--google--gemma-2b/snapshots/2ac59a5d7bf4e1425010f0d457dde7d146658953",
            "r1-qwen-7b": "/nas/shared/NLP_A100/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/393119fcd6a873e5776c79b0db01c96911f5f0fc/",
            "r1-llama-8b": "/nas/shared/NLP_A100/hf_hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-8B/snapshots/81cee02dd020268dced5fa1327e8555acce9c63c/"
        }
        if self.args.model_id not in model_paths:
            raise ValueError(f"Unknown model_id: {self.args.model_id}")
        return model_paths[self.args.model_id]

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
            "strategy": "Please solve the following problem step by step.\nYou will be presented with one question. At the end, you must output 'Yes' or 'No' after 'The answer is: '.",
            "cs": "Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are five options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response.",
            "gpqa": "Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response.",
            "arc": "Please solve the following problem step by step.\nYou will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response.",
            "scibench": "Please solve the following problem step by step.\nWhen you reach the answer, please include the answer in the box format and finish the reasoning with <end_of_reasoning>.",
            "truthfulqa_mc1": "Please solve the following problem step by step.\nYou will be presented with a question. There are multiple options to be chosen from, you need to choose the only correct option to answer that question.\nPlease output the predicted option after 'The answer is: ' at the end of the response.",
            "humaneval": "Please directly complete the following code without any additional comments.\n"
        }

        # 特殊处理 r1 模型
        if hasattr(self.args, 'model_id') and 'r1' in self.args.model_id:
            return "A conversation between user and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."

        return prompts.get(dataset_type, "")

    def generate_responses(self, inputs, num_samples=1):
        """
        Generate responses using the language model
        Args:
            inputs: Input prompts
            num_samples: Number of responses to generate per input
        Returns:
            Model outputs
        """
        sampling_params = SamplingParams(
            max_tokens=1024,
            n=num_samples,
            logprobs=0,
            temperature=0.6,
            stop=["<end_of_reasoning>"]
        )
        return self.model.generate(inputs, sampling_params)

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
        """
        Select final response based on strategy
        Args:
            responses: List of response texts
            logprobs: Log probabilities for each response
            advantages: Advantage values for each response
        Returns:
            Index of selected response
        """
        if self.args.strategy == "cluster":
            clusters, info = self.cluster_responses(responses, advantages)
            if info["state"] != "success":
                # Fallback to advantage-based selection
                weights = softmax([adv/TEMPERATURE for adv in advantages])
                return np.random.choice(len(advantages), p=weights, replace=False)

            # Select from largest cluster
            largest_cluster = max(clusters, key=len)
            cluster_advantages = [advantages[i] for i in largest_cluster]
            weights = softmax([adv/TEMPERATURE for adv in cluster_advantages])
            selected_idx = np.random.choice(len(weights), p=weights)
            return largest_cluster[selected_idx]

        elif self.args.strategy in ["sir", "sir_no_replace"]:
            weights = softmax([logp/TEMPERATURE for logp in logprobs])
            return np.random.choice(len(weights), p=weights,
                                    replace="no_replace" not in self.args.strategy)

        elif self.args.strategy in ["adv", "adv_no_replace"]:
            weights = softmax([adv/TEMPERATURE for adv in advantages])
            return np.random.choice(len(weights), p=weights,
                                    replace="no_replace" not in self.args.strategy)

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

        # Multi-step reasoning
        for step in range(self.args.num_foresight):
            step_results = self._process_step(
                example,
                system_prompt,
                previous_steps,
                previous_values,
                token_stats,
                rollout_stats
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
            rollout_stats
        )

        return {
            "response": final_result["response"],
            "token_stats": token_stats,
            "rollout_stats": rollout_stats,
            "trajectories": {
                "steps": step_pool,
                "probs": prob_pool,
                "advantages": adv_pool,
                "final": final_result["trajectories"]
            }
        }

    def _process_step(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats):
        """Process a single reasoning step"""
        # Generate responses for each beam
        all_responses = []
        all_logprobs = []
        all_advantages = []

        for beam_idx in range(self.args.step_beam_size):
            # Prepare input with previous steps
            chat = self._prepare_chat_template(example, system_prompt)
            chat[-1]["content"] = previous_steps[beam_idx]

            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            # Track input tokens
            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])

            # Generate responses
            outputs = self.generate_responses(
                [inputs],
                num_samples=self.args.num_rollout
            )
            rollout_stats["total"] += self.args.num_rollout

            # Process outputs
            beam_responses = []
            beam_logprobs = []
            beam_advantages = []

            for output in outputs[0].outputs:
                response = output.text.strip()
                logprob = output.cumulative_logprob / \
                    (len(output.token_ids) + 1e-8)
                advantage = logprob - previous_values[beam_idx]

                beam_responses.append(response)
                beam_logprobs.append(logprob)
                beam_advantages.append(advantage)

                token_stats["output"] += len(output.token_ids)

            all_responses.extend(beam_responses)
            all_logprobs.extend(beam_logprobs)
            all_advantages.extend(beam_advantages)

        # Select best responses
        selected_indices = self._select_step_responses(
            all_responses,
            all_logprobs,
            all_advantages
        )

        # Prepare next steps
        next_steps = []
        next_values = []
        for idx in selected_indices:
            beam_idx = idx // self.args.num_rollout
            response_idx = idx % self.args.num_rollout

            next_step = previous_steps[beam_idx] + all_responses[idx] + "\n"
            next_steps.append(next_step)
            next_values.append(all_logprobs[idx])

        return {
            "next_steps": next_steps,
            "next_values": next_values,
            "trajectories": all_responses,
            "steps": selected_indices,
            "logprobs": all_logprobs,
            "advantages": all_advantages
        }

    def _select_step_responses(self, responses, logprobs, advantages):
        """Select responses for next step based on strategy"""
        if self.args.strategy == "cluster":
            clusters, info = self.cluster_responses(responses, advantages)
            if info["state"] != "success":
                # Fallback to advantage-based selection
                weights = softmax([adv/TEMPERATURE for adv in advantages])
                return np.random.choice(
                    len(advantages),
                    size=self.args.step_beam_size,
                    p=weights,
                    replace=False
                ).tolist()

            # Select from largest cluster
            largest_cluster = max(clusters, key=len)
            cluster_advantages = [advantages[i] for i in largest_cluster]
            weights = softmax([adv/TEMPERATURE for adv in cluster_advantages])

            selected = np.random.choice(
                len(weights),
                size=min(self.args.step_beam_size, len(weights)),
                p=weights,
                replace=False
            )
            return [largest_cluster[i] for i in selected]

        elif "sir" in self.args.strategy:
            weights = softmax([logp/TEMPERATURE for logp in logprobs])
            return np.random.choice(
                len(weights),
                size=self.args.step_beam_size,
                p=weights,
                replace="no_replace" not in self.args.strategy
            ).tolist()

        elif "adv" in self.args.strategy:
            weights = softmax([adv/TEMPERATURE for adv in advantages])
            return np.random.choice(
                len(weights),
                size=self.args.step_beam_size,
                p=weights,
                replace="no_replace" not in self.args.strategy
            ).tolist()

        else:
            raise ValueError(f"Unknown strategy: {self.args.strategy}")

    def _should_stop_early(self, step_results, current_step):
        """Check if early stopping conditions are met"""
        if current_step < self.args.least_foresight_num:
            return False

        if self.args.depth_pruning_strategy == "cluster":
            # Check if responses are becoming similar
            responses = step_results["trajectories"]

            # Skip if not enough valid responses
            valid_responses = [r for r in responses if r.strip()]
            if len(valid_responses) < self.args.step_beam_size:
                return False

            try:
                # Cluster responses
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(valid_responses)
                kmeans = KMeans(n_clusters=2)
                kmeans.fit(X)

                # Check cluster sizes
                labels = kmeans.labels_
                sizes = np.bincount(labels)
                largest_ratio = max(sizes) / len(valid_responses)

                return largest_ratio > self.args.threshold

            except Exception:
                return False

        return False

    def _generate_final_response(self, example, system_prompt, previous_steps, previous_values, token_stats, rollout_stats):
        """Generate final response after multi-step reasoning"""
        # Generate responses for each beam
        all_responses = []
        all_logprobs = []
        all_advantages = []

        for beam_idx in range(self.args.step_beam_size):
            chat = self._prepare_chat_template(example, system_prompt)
            chat[-1]["content"] = previous_steps[beam_idx]

            inputs = self.tokenizer.apply_chat_template(
                chat,
                tokenize=False
            ).rstrip(self.tokenizer.eos_token).rstrip()

            token_stats["input"] += len(self.tokenizer(inputs)["input_ids"])

            outputs = self.generate_responses(
                [inputs],
                num_samples=1  # Only one response per beam for final step
            )
            rollout_stats["total"] += 1

            output = outputs[0].outputs[0]
            response = output.text.strip()
            logprob = output.cumulative_logprob / \
                (len(output.token_ids) + 1e-8)
            advantage = logprob - previous_values[beam_idx]

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

        return {
            "response": previous_steps[selected_idx] + all_responses[selected_idx],
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


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Phi-Decoding Algorithm")

    # Model configuration
    parser.add_argument('--model_id', type=str, default='r1-qwen-7b',
                        help='Model identifier')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')

    # Data configuration
    parser.add_argument('--datasets', type=str, default='aime',
                        help='Dataset type')
    parser.add_argument('--data_path', type=str,
                        default='/cpfs01/user/xufangzhi/o1/data/aime2024_test.json',
                        help='Path to input data')
    parser.add_argument('--output_dir', type=str,
                        default='/cpfs01/user/xufangzhi/o1/cluster_results/',
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
    parser.add_argument('--cluster_num', type=int, default=2,
                        help='Number of clusters for clustering strategy')
    parser.add_argument('--threshold', type=float, default=0.79,
                        help='Threshold for early stopping')
    parser.add_argument('--least_foresight_num', type=int, default=4,
                        help='Minimum number of foresight steps')

    # Execution configuration
    parser.add_argument('--record_process', type=bool, default=True,
                        help='Whether to record the decoding process')
    parser.add_argument('--file_name', type=str, default='test_3',
                        help='Output file name')
    parser.add_argument('--time_path', type=str,
                        default='/cpfs01/user/xufangzhi/o1/cluster_results/time/',
                        help='Path to save timing information')

    return parser.parse_args()


def main():
    """Main execution function"""
    # Parse arguments
    args = parse_arguments()

    # Initialize decoder
    decoder = PhiDecoder(args)

    # Load test data
    with open(args.data_path) as f:
        test_data = json.load(f)

    # Setup output paths
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.file_name}.json")

    # Track statistics
    start_time = time.time()
    total_stats = {
        "total_rollouts": 0,
        "saved_rollouts": 0,
        "input_tokens": 0,
        "output_tokens": 0
    }

    # Process each test example
    results = []
    for i, example in enumerate(test_data):
        try:
            # Generate system prompt
            system_prompt = decoder.get_system_prompt(args.datasets)

            # Process example
            result = decoder.process_example(example, system_prompt)

            # Update statistics
            total_stats["total_rollouts"] += result["rollout_stats"]["total"]
            total_stats["saved_rollouts"] += result["rollout_stats"]["saved"]
            total_stats["input_tokens"] += result["token_stats"]["input"]
            total_stats["output_tokens"] += result["token_stats"]["output"]

            # Save result
            results.append({
                "id": i,
                "question": example["input"],
                "ground_truth": example.get("target"),
                "response": result["response"],
                "trajectories": result["trajectories"] if args.record_process else None
            })

        except Exception as e:
            print(f"Error processing example {i}: {e}")
            continue

        # Save results
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

    # Save timing information
    end_time = time.time()
    time_span = end_time - start_time

    timing_info = {
        "time": time_span,
        "statistics": total_stats,
        "config": vars(args)
    }

    with open(os.path.join(args.time_path, f"{args.file_name}.json"), "w") as f:
        json.dump(timing_info, f, indent=2)


if __name__ == "__main__":
    main()
