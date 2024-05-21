import os
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertModel
import datetime
import argparse
import logging
from transformers import pipeline
import datasets
import ppl_mcts

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

def perpare_datasets():
    dataset_sst2 = datasets.load_dataset("sst2", split="validation")
    dataset_sst5 = datasets.load_dataset("SetFit/sst5", split="validation")

    neg2pot1_sst2 = dataset_sst2.filter(lambda x: x["label"] == 0)
    pot = dataset_sst2.filter(lambda x: x["label"] == 1)
    pot_len = len(neg2pot1_sst2) // 2
    for i in range(pot_len):
        neg2pot1_sst2 = neg2pot1_sst2.add_item(pot[i])

    neg2pot1_sst5 = dataset_sst5.filter(lambda x: x["label"] == 0)
    pot = dataset_sst5.filter(lambda x: x["label"] == 4)
    pot_len = len(neg2pot1_sst5) // 2
    for i in range(pot_len):
        neg2pot1_sst5 =neg2pot1_sst5.add_item(pot[i])

    return neg2pot1_sst2, neg2pot1_sst5


def ppl_greedy(prompt, generate_length, var_threshold, top_k):
    input = tokenizer(prompt, return_tensors="pt", padding=True)
    usage_mcts = 0
    for i in range(generate_length):
        output = model(**input, return_dict=True)
        logits = output.logits[:, -1, :].detach().numpy()
        top_k_idx = np.argpartition(logits, -top_k, axis=-1)[0, -top_k:]
        top_k_logits = logits[:, top_k_idx]
        # print(top_k_logits.mean(), top_k_logits.var())
        if top_k_logits.var() > var_threshold:
            next_id = top_k_idx[np.argmax(top_k_logits)]
            input['input_ids'] = torch.cat((input['input_ids'], torch.tensor([[next_id]])), dim=1)
            input['attention_mask'] = torch.cat((input['attention_mask'], torch.tensor([[1]])), dim=1)
            # print(tokenizer.batch_decode(input.input_ids))
        else:
            newtext = ppl_mcts.mcts_search(tokenizer.batch_decode(input.input_ids))
            input = tokenizer(newtext, return_tensors="pt", padding=True)
            usage_mcts += 1
            # print(newtext)
    return tokenizer.batch_decode(input.input_ids), usage_mcts / generate_length


def make_save_path(targetFile="result1"):
    dir = f"./{targetFile}/" + str(datetime.datetime.now().strftime('%m-%d-h%H'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/"


if __name__ == "__main__":
    save_path = make_save_path()
    sst2, sst5 = perpare_datasets()
    judge = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    prompt1 = "Please continue this sentence and make it positive : \n"
    prompt2 = "\n Your result is: " 
    pos2neg, neg2pos, p2p, n2n = 0, 0, 0, 0
    total = len(sst2)
    sst2_pbar = tqdm(total = total, desc="SST2 process")
    sst2_freq = []
    sst2_time = []
    for p in sst2:
        start_time = datetime.datetime.now()
        gen_text = ppl_greedy([prompt1 + p["sentence"][0] + prompt2], 50, 1, 10)
        end_time = datetime.datetime.now()
        judgement = judge(gen_text[0])
        if judgement[0]['label'] == 'Positive':
            pred = 1
            neg2pos += 1 if p["label"] == 0 else 0
            p2p += 1 if p["label"] == 1 else 0
        else:
            pred = 0
            pos2neg += 1 if p["label"] == 1 else 0
            n2n += 1 if p["label"] == 0 else 0
        sst2_freq.append(gen_text[1])
        sst2_time.append((end_time - start_time).seconds)
        with open(save_path + "sst2_ppl-greedy.jsonl", "a") as fw:
            fw.write(json.dumps({"text": gen_text, "real_label": p["label"], "pred_label": pred, "usage_ppl": gen_text[1], "usage_time": (end_time - start_time).seconds}))
            fw.write("\n")
        sst2_pbar.update(1)

    with open(save_path + "sst2_ppl-greedy.jsonl", "a") as fw:
        fw.write(json.dumps({"pos2neg": pos2neg,  "neg2pos": neg2pos, "p2p": p2p, "n2n": n2n}))
        fw.write("\n")
        fw.write(json.dumps({"total": total}))
        fw.write("\n")
        fw.write(json.dumps({"sst2_freq": sum(sst2_freq) / len(sst2_freq)}))
        fw.write("\n")
        fw.write(json.dumps({"sst2_time": sum(sst2_time) / len(sst2_time)}))
        fw.write("\n")

    pos2neg, neg2pos, p2p, n2n = 0, 0, 0, 0
    total = len(sst5)
    sst5_pbar = tqdm(total = total, desc="SST5 process")
    sst5_freq = []
    sst5_time = []
    for p in sst5:
        start_time = datetime.datetime.now()
        gen_text = ppl_greedy([prompt1 + p["sentence"][0] + prompt2], 50, 1, 10)
        end_time = datetime.datetime.now()
        judgement = judge(gen_text[0])
        if judgement[0]['label'] == 'Positive':
            pred = 1
            neg2pos += 1 if p["label"] == 0 else 0
            p2p += 1 if p["label"] == 1 else 0
        else:
            pred = 0
            pos2neg += 1 if p["label"] == 1 else 0
            n2n += 1 if p["label"] == 0 else 0

        sst5_freq.append(gen_text[1])
        sst5_time.append((end_time - start_time).seconds)
        with open(save_path + "sst5_ppl-greedy.jsonl", "a") as fw:
            fw.write(json.dumps({"text": gen_text, "real_label": p["label"], "pred_label": pred, "usage_ppl": gen_text[1], "usage_time": (end_time - start_time).seconds}))
            fw.write("\n")
        sst5_pbar.update(1)
            
    with open(save_path + "sst2_ppl-greedy.jsonl", "a") as fw:
        fw.write(json.dumps({"pos2neg": pos2neg,  "neg2pos": neg2pos, "p2p": p2p, "n2n": n2n}))
        fw.write("\n")
        fw.write(json.dumps({"total": total}))
        fw.write("\n")
        fw.write(json.dumps({"sst5_freq": sum(sst5_freq) / len(sst5_freq)}))
        fw.write("\n")
        fw.write(json.dumps({"sst5_time": sum(sst5_time) / len(sst5_time)}))
        fw.write("\n")

# print(tokenizer.batch_decode(model.generate(**tokenizer(text, return_tensors="pt", padding=True), do_sample=False)))