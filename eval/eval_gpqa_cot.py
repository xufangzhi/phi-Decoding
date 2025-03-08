import os
import json
import numpy as np
import re
import random

def getAnswer(response):
    pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char

    return ""

prediction = []
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gpqa_diamond_test_241210-1_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gpqa_main_test_qwen2.5_sir_no_replace_rollout_0_foresight_0.json") as file:
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/gpqa_main_test_250204-2_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/250130-76.json") as file:
    for line in file:
        prediction.append(json.loads(line))
# prediction = prediction[:223]
print(len(prediction))

# print(prediction[0])

correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response']
    # print(response)
    pred = getAnswer(response)
    gt = prediction[i]['ground_truth']
    try:
        if gt==pred:
            correct_num += 1
    except:
        continue
print(correct_num/len(prediction))