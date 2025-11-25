import sys 
import json
from tokenization import NllbTokenizer
from bytetokenizer import ByteTokenizer
from collections import Counter
from tqdm import tqdm

model_used = False
if sys.argv[1] == "token":
    tokenizer = NllbTokenizer("600M")
    model_used = True
elif sys.argv[1] == "byte":
    tokenizer = ByteTokenizer() 
    
token_counts = Counter() 

if sys.argv[2] == "1b":
    years = ["07", "08", "09", "10", "11"]
    for year in years: 
        src_path = f"corpus/training-monolingual/news.20{year}.en.shuffled"
        with open(src_path) as reader:
            for line in tqdm(reader): 
                tokens = tokenizer(line.strip())
                if model_used: 
                    token_counts.update(tokens["input_ids"].flatten().tolist())
                else:
                    token_counts.update(tokens["input_ids"])

elif sys.argv[2] == "x": 
    with open("/mnt/storage/hopkins/tweets.txt") as reader: 
        for line in tqdm(reader):
            tweet = json.loads(line)["content"]
            tokens = tokenizer(tweet.strip())
            if model_used:
                token_counts.update(tokens["input_ids"].flatten().tolist())
            else:
                token_counts.update(tokens["input_ids"])

fname = f"{sys.argv[2]}_{sys.argv[1]}"
with open(f"unigram_lms/{fname}unigram_lm.txt", "a") as file: 
    file.write(str(token_counts))
