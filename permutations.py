import random
from pathlib import Path
import json
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_constant_schedule_with_warmup,
)

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
    

#function to save dictionary into json file
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


#function to read json file
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


def batch_sort(batch_size=128):
    OUT_DIR = Path("./optimized_data")
    OUT_DIR.mkdir(exist_ok=True)
    num_lines = 0
    line_list_en = []
    with open("europarlData/train.en", "r", encoding="utf-8") as f:
        line_list_en = f.readlines()
        for line in line_list_en:
            num_lines += 1

    number_of_batches = num_lines//batch_size
    base_model = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.src_lang = "eng_Latn"
    line_length_dict = {} #Key: line number; Value: line length
    i = 0
    for line in line_list_en:
        tokenized_line = tokenizer(line)['input_ids']
        line_length_dict[i] = len(tokenized_line)
        i+=1
    sorted_lines = sorted(line_length_dict.items(), key=lambda item: item[1])
    order_of_lines = []
    for pair in sorted_lines:
        order_of_lines.append(pair[0])
    batch_list = []
    for i in range(number_of_batches):
        this_batch = []
        for j in range(batch_size):
            this_batch.append(order_of_lines[i*batch_size+j])
        batch_list.append(this_batch)
    pmap_batches = CreateRandomPermutationWithFixedPoints(number_of_batches,[]) 
    reshuffled_batches = []
    for k in range(number_of_batches):
        reshuffled_batches.append(batch_list[pmap_batches(k)])
    with open(OUT_DIR / f"optimized_train_{batch_size}.en","w") as file:
        for i in range(number_of_batches):
            for j in range(batch_size):
                file.write(line_list_en[reshuffled_batches[i][j]])

    LANGS = [
    "bg", "cs", "da", "de", "el",
    "es", "et", "fi", "fr", "hu",
    "it", "lt", "lv", "nl", "pl",
    "pt", "ro", "sk", "sl", "sv",
    ]
    for lang_code in LANGS:
        line_list = []
        with open(f"europarlData/train.{lang_code}", "r") as f:
            line_list = f.readlines()
        
        with open(OUT_DIR / f"optimized_train_{batch_size}.{lang_code}","w") as file:
            for i in range(number_of_batches):
                for j in range(batch_size):
                    file.write(line_list[reshuffled_batches[i][j]])
    

