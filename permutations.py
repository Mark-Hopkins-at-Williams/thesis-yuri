import random
from pathlib import Path
import json
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)
import math

class CreateRandomPermutationWithFixedPoints:
    def __init__(self, num:int, fixed_points):
        self.num = num
        self.fixed_points = fixed_points
        non_fixed_points = [i for i in list(range(num)) if i not in fixed_points]
        self.result_dict = {}
        self.inverse_dict = {}
        for i in range(num):
            if i in fixed_points:
                self.result_dict[i] = i
                self.inverse_dict[i] = i
            else:
                self.result_dict[i] = random.choice(non_fixed_points)
                non_fixed_points.remove(self.result_dict[i])
                self.inverse_dict[self.result_dict[i]] = i        
        
    def __call__(self, num):
        return self.result_dict.get(num, num)
    
    def get_inverse(self):
        q = CreateRandomPermutationWithFixedPoints(self.num, self.fixed_points)
        q.result_dict = self.inverse_dict
        q.inverse_dict = self.result_dict
        return q
    

# Function to save dictionary into json file
def save_permutation_map(pmap, filename):
    data = {}
    
    for key,value in pmap.items():
        data[key] = {
            "fixed points": value.fixed_points,
            "rng": value.num,
            "result dictionary": value.result_dict,
            "inverse dictionary": value.inverse_dict
            }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Function to read json file
def load_permutation_map(filename):
    with open(filename, 'r') as f:
        data_dict = json.load(f)
        data = {}
        for key,value in data_dict.items():
            obj = CreateRandomPermutationWithFixedPoints(
                num = value["rng"],
                fixed_points = value["fixed points"]
            )
            obj.result_dict = {int(i):s for i, s in value["result dictionary"].items()}
            obj.inverse_dict = {int(i):s for i, s in value["inverse dictionary"].items()}
            data[key] = obj
        return data

