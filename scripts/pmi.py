import math
from transformers import AutoTokenizer

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def pmi(token_num, filename1, filename2, tokenizer):

    line_list1 = []
    line_list2 = []
    token = tokenizer.convert_ids_to_tokens([token_num])[0]
    token_clean = tokenizer.convert_ids_to_tokens([token_num])[0].replace('▁','')
    
    with open(filename1, "r") as f1:
        line_list1 = f1.readlines()
    with open(filename2, "r") as f2:
        line_list2 = f2.readlines()
    in_first = 0
    in_second = 0
    in_both = 0
    
    for i in range(len(line_list1)):
        tokenized1 = tokenizer(line_list1[i])
        tokenized2 = tokenizer(line_list2[i])
        if token_num in tokenized1['input_ids']:
            in_first += 1
            if token_num in tokenized2['input_ids']:
                in_second += 1
                in_both += 1

        elif token_num in tokenized2['input_ids']:
            in_second += 1
    if in_second == 0 or in_first == 0:
        return "Token does not appear in both lang"
    r = (in_both*len(line_list1))/(in_second*in_first)
    
    log_r = math.log2(r)
   
    return log_r

r = pmi(200251,"../europarlData/dev.da", "../europarlData/dev.de", tokenizer)
print(r)