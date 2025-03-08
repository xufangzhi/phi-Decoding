import os
import json
import numpy as np
import re

from collections import Counter

def most_common_element(lst):
    counts = Counter(lst)
    max_count = max(counts.values())
    most_common = [k for k, v in counts.items() if v == max_count][0]
    return most_common


def extract_numbers(input_string):
    return re.findall(r'\d+', input_string)


def find_last_number(input_string):
    words = input_string.split()
    numbers = []
    for word in words:
        try:
            number = float(word)
            numbers.append(number)
        except ValueError:
            pass

    if not numbers:
        return ""
    return str(numbers[-1])


def eval_cot_answer(pred, gt):
    boxed_contents = ""
    try:
        if "\\boxed{" in pred:
            boxed_contents = re.findall(r'\\boxed\{(.*?)\}', pred)
            if boxed_contents:
                boxed_contents = boxed_contents[-1]
            else:
                boxed_contents = ""
        elif "<boxed>" in pred:
            boxed_contents = re.findall(r'<boxed>(\d+)<\/boxed>', pred)
            if boxed_contents:
                boxed_contents = boxed_contents[-1].strip()
            else:
                boxed_contents = ""
        elif "answer is:" in pred:
            boxed_contents = pred.split("answer is:")[-1].strip()
        else:
            boxed_contents = find_last_number(pred[-50:])  # from last 100 chars to parse the possible answer
    except:
        return False, None
    
    answer = boxed_contents.strip('\\').replace(",","").strip("$").strip()

    if "." in answer:
        pred_ans = answer
    elif "frac" in answer and len(extract_numbers(answer))==2:
        if float(extract_numbers(answer)[1]) != 0:
            pred_ans = float(extract_numbers(answer)[0]) / float(extract_numbers(answer)[1])
        else:
            return False, None
    elif extract_numbers(answer):
        pred_ans = extract_numbers(answer)[0]
    else:
        return False, None

    try:
        if abs(float(gt) - float(pred_ans)) < 1e-5:
            return True, pred_ans
        else:
            return False, pred_ans
    except:
        return False, None
    return False, None


prediction = []
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gsm_test_241110-1.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gsm_test_qwen2.5_ar.json") as file:
# 757
# with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/250131-23.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gsm_test_adv_no_replace_step_beam_4_rollout_4_foresight_4.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gsm_test_250204-2_sir_no_replace_rollout_0_foresight_0.json") as file:
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/math_500_test_250204-2_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/math_test_250126-3_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/math_500_test_llama3.1_sir_rollout_0_foresight_0_beam4.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/math_500_test_llama3.1_sir_no_replace_rollout_0_foresight_0.json") as file:

    for line in file:
        prediction.append(json.loads(line))
# prediction = prediction[:640]
print(len(prediction))


correct_num = 0
correct_num_passk = 0
correct_num_sc = 0
for i in range(len(prediction)):
    # preprocess
    if eval_cot_answer(prediction[i]['response'], prediction[i]['ground_truth'])[0]:
        correct_num += 1

    correct_passk = False
    candidate_list = []
    if "response_all_beams" in prediction[i]:
        for j in range(len(prediction[i]['response_all_beams'])):
            # calculate pass@K
            if eval_cot_answer(prediction[i]['response_all_beams'][j], prediction[i]['ground_truth'])[0]:
                    correct_passk = True

            # calculate sc@K
            if eval_cot_answer(prediction[i]['response_all_beams'][j], prediction[i]['ground_truth'])[1]:
                candidate_list.append(float(eval_cot_answer(prediction[i]['response_all_beams'][j], prediction[i]['ground_truth'])[1]))

        if correct_passk:
            correct_num_passk += 1

        try:
            pred = most_common_element(candidate_list)
            if abs(float(prediction[i]['ground_truth']) - float(pred)) < 1e-5:
                correct_num_sc += 1
        except:
            continue

print("Acc: ", correct_num/len(prediction))

print("Pass@K: ", correct_num_passk / len(prediction))

print("SC@K: ", correct_num_sc / len(prediction))