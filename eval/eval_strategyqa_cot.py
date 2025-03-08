import os
import json
import numpy as np
import re


def getAnswer(response):
    pred = response.split("answer is")[-1]
    if "Yes" in pred:
        return "Yes"
    elif "No" in pred:
        return "No"
    return ""


prediction = []
# with open(f"/cpfs01/user/xufangzhi/o1/cluster_results/241228-13.json") as file:
with open(f"/cpfs01/user/xufangzhi/o1/infer/results/strategyqa_test_241202-3_sir_no_replace_rollout_0_foresight_0.json") as file:
# with open(f"/cpfs01/user/xufangzhi/o1/infer/results/strategyqa_test_qwen2.5_sir_no_replace_rollout_0_foresight_0.json") as file:
    for line in file:
        prediction.append(json.loads(line))

print(len(prediction))



correct_num = 0
for i in range(len(prediction)):
    response = prediction[i]['response'].strip()
    
    pred = getAnswer(response)
    # print(response)
    # print(pred)
    # input()
    # print("====")
    gt = prediction[i]['ground_truth']

    try:
        if gt[pred]==1:
            correct_num += 1
    except:
        continue

print(correct_num/len(prediction))