import os
import json
import numpy as np
import re

def getAnswer(response):
    pred = response.split("The answer is:")[-1]
    for char in pred:
        if char.isupper() and char in ["A", "B", "C", "D", "E"]:
            return char
    return ""

prediction = []
with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/241228-14.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/csqa_test_241229-3_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/csqa_test_llama3.1_sir_no_replace_rollout_0_foresight_0.json") as file:
    for line in file:
        prediction.append(json.loads(line))

print(len(prediction))



correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response']
    
    pred = getAnswer(response)
    gt = prediction[i]['ground_truth']
    try:
        if gt==pred:
            correct_num += 1
    except:
        continue

print(correct_num/len(prediction))