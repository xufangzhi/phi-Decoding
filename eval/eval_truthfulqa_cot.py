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
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/truthfulqa_mc1_test_llama3.1_sir_no_replace_rollout_0_foresight_0.json") as file:
    for line in file:
        prediction.append(json.loads(line))
# prediction = prediction[:250]
print(len(prediction))

# map_dict = {
#     "A": 0,
#     "B": 1,
#     "C": 2,
#     "D": 3,
# }

correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response']
    pred = find_last_uppercase(response)
    gt = prediction[i]['ground_truth']
    try:
        if pred==gt:
            correct_num += 1
    except:
        continue

print(correct_num/len(prediction))