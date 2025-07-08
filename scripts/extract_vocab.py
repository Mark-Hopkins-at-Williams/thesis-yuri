
from pathlib import Path
from transformers import AutoTokenizer 
import math

def extract_vocab(filename, lang_code1, lang_code2, filter_num, tokenizer):
    OUT_DIR = Path(f"../pmi_lang_pairs_data/{filter_num}filtered")
    OUT_DIR.mkdir(exist_ok=True)
    

    pmi_values = []
    token_list = []
    token_pmi_dict = {}

    with open(OUT_DIR / filename, "r", encoding="utf-8") as f:
        for line in f:
            if "PMI:" in line:
                try:
                    token_part, pmi_part = line.strip().split("PMI:")
                    token = token_part.strip()
                    pmi = float(pmi_part.strip())
                    pmi_values.append(pmi)
                    token_list.append(token)
                    token_pmi_dict[token] = pmi
                except ValueError:
                    continue


    candidates = [] #(token1, token2)

    i = 0
    k = 1
    patience = 0.02
    while i < len(pmi_values) - 1:
        j = i+k if i+k<len(pmi_values) else len(pmi_values)
        if (pmi_values[i] - pmi_values[j]) < patience:
            candidates.append((token_list[i],token_list[j]))
            k += 1
        else:
            k = 1
            i += 1
    print(candidates)
    line_list1 = []
    line_list2 = []

    filename1 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/train.{lang_code1}"
    filename2 = f"/mnt/storage/sotnichenko/encoder-decoder-finetuning/europarlData/train.{lang_code2}"
    with open(filename1, "r") as f1:
        line_list1 = f1.readlines()

    with open(filename2, "r") as f2:
        line_list2 = f2.readlines()

    print("Read files")
    candidate_analysis = {} #Key: (token1,token2); Value: (pmi_pair_lang1, pmi_pair_lang2)
    lang1_token_counter = {} #Key: (token1,token2); Value: (token1_count_lang1, token2_count_lang1, both_tokens)
    lang2_token_counter = {} #Key: (token1,token2); Value: (token1_count_lang2, token2_count_lang2, both_tokens)
    for i in range(len(line_list1)):
        print(i)
        tokenized1 = tokenizer(line_list1[i])
        tokenized2 = tokenizer(line_list2[i])
        for candidate_pair in candidates:
            token1 = tokenizer.convert_tokens_to_ids(candidate_pair[0])
            token2 = tokenizer.convert_tokens_to_ids(candidate_pair[1])
            if (token1,token2) not in lang1_token_counter.keys():
                lang1_token_counter[(token1,token2)] = [0,0,0]
                lang2_token_counter[(token1,token2)] = [0,0,0]
            
            if token1 in tokenized1['input_ids']:
                if token2 in tokenized1['input_ids']:
                    lang1_token_counter[(token1,token2)][0] += 1
                    lang1_token_counter[(token1,token2)][1] += 1
                    lang1_token_counter[(token1,token2)][2] += 1
            elif token2 in tokenized1['input_ids']:
                lang1_token_counter[(token1,token2)][1] += 1
            
            if token1 in tokenized2['input_ids']:
                if token2 in tokenized1['input_ids']:
                    lang2_token_counter[(token1,token2)][0] += 1
                    lang2_token_counter[(token1,token2)][1] += 1
                    lang2_token_counter[(token1,token2)][2] += 1
            elif token2 in tokenized1['input_ids']:
                lang2_token_counter[(token1,token2)][1] += 1

            if i == len(line_list1) - 1:
                        print(lang1_token_counter)
                        token1_count_lang1 = lang1_token_counter[(token1,token2)][0]
                        token2_count_lang1 = lang1_token_counter[(token1,token2)][1]
                        both_tokens_lang1 = lang1_token_counter[(token1,token2)][2]
                        r1 = (both_tokens_lang1*len(line_list1))/(token1_count_lang1*token2_count_lang1) if (token1_count_lang1 != 0 and token2_count_lang1 !=0) else 0
                        candidate_pair_pmi_lang1 = math.log2(r1) if r1 > 0 else 0
                        candidate_analysis[(token1,token2)] = [candidate_pair_pmi_lang1,0]

                        token1_count_lang2 = lang2_token_counter[(token1,token2)][0]
                        token2_count_lang2 = lang2_token_counter[(token1,token2)][1]
                        both_tokens_lang2 = lang2_token_counter[(token1,token2)][2]
                        r2 = (both_tokens_lang2*len(line_list1))/(token1_count_lang2*token2_count_lang2) if (token1_count_lang2 != 0 and token2_count_lang2 !=0) else 0
                        candidate_pair_pmi_lang2 = math.log2(r2) if r2 > 0 else 0
                        candidate_analysis[(token1,token2)][1] += candidate_pair_pmi_lang2
    finalists = []
    print(candidate_analysis)
    for candidate, pmi_pair in candidate_analysis.items():
        if (pmi_pair[0]+pmi_pair[1]) >= 3:
            finalists.append(tokenizer.convert_ids_to_tokens(candidate[0])+tokenizer.convert_ids_to_tokens(candidate[1]))


    return finalists

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(extract_vocab("es_en_pmi_ranking.txt", "es", "en", 200, tokenizer))
