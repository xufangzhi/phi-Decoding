import os
import json
import numpy as np
import re


def find_last_uppercase(input_str):
    input_str = input_str.split("answer is")[-1]
    for char in input_str:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char
    return None


prediction = []
# with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/250131-14.json") as file:
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/logiqa_test_250204-2_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/reclor_val_250204-2_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/reclor_val_llama3.1.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/reclor_val_llama3.1.json") as file:
    for line in file:
        prediction.append(json.loads(line))
# prediction = prediction[:378]
print(len(prediction))

map_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response']
    pred = find_last_uppercase(response)
    gt = prediction[i]['ground_truth']
    try:
        if map_dict[pred]==gt:
            correct_num += 1
    except:
        # print('fail')
        continue

print(correct_num/len(prediction))