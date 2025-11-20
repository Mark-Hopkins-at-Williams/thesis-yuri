from bytetokenizer import ByteTokenizer
from collections import Counter
from tqdm import tqdm


tokenizer = ByteTokenizer()
token_counts = Counter()
with open('/mnt/storage/yuri/thesis-yuri/corpus/training-monolingual/news.2007.en.shuffled') as reader:
    for line in tqdm(reader):
        tokens = tokenizer(line.strip())
        token_counts.update(tokens['input_ids'])
    


        
print(token_counts)