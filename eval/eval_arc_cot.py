import os
import json
import numpy as np
import re

def getAnswer(response):
    pred = response.split("answer is")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D"]:
            return char
    
    return ""

prediction = []
idx  = 0
# with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/250131-21.json") as file:
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/arc-c_test_241202-3_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/arc-c_test_qwen2.5_sir_no_replace_rollout_0_foresight_0.json") as file:
    for line in file:
        idx += 1
        try:
            prediction.append(json.loads(line))
        except:
            print(idx, ' error')
# prediction = prediction[:581]
print(len(prediction))



correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response']
    pred = getAnswer(response)
    # print(response)
    # print(pred)
    # input()
    # print("==========")
    gt = prediction[i]['ground_truth']
    try:
        if gt==pred:
            correct_num += 1
    except:
        continue

print(correct_num/len(prediction))