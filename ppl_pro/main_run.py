import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import json
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertModel
import datetime
from transformers import pipeline
import datasets
import ipdb
import arguments
import ppl_mcts

args = arguments.get_args()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

def perpare_datasets_sst2():
    dataset_sst2 = datasets.load_dataset("sst2", split="validation")

    neg2pot1_sst2 = dataset_sst2.filter(lambda x: x["label"] == 0)
    pot = dataset_sst2.filter(lambda x: x["label"] == 1)
    pot_len = len(neg2pot1_sst2) // 2
    for i in range(pot_len):
        neg2pot1_sst2 = neg2pot1_sst2.add_item(pot[i])

    return neg2pot1_sst2


def perpare_datasets_sst5():
    dataset_sst5 = datasets.load_dataset("SetFit/sst5", split="validation")

    neg2pot1_sst5 = dataset_sst5.filter(lambda x: x["label"] == 0)
    pot = dataset_sst5.filter(lambda x: x["label"] == 4)
    pot_len = len(neg2pot1_sst5) // 2
    for i in range(pot_len):
        neg2pot1_sst5 =neg2pot1_sst5.add_item(pot[i])
    # transfer label 4 to 1
    neg2pot1_sst5 = neg2pot1_sst5.map(lambda x: {'label': 1 if x["label"] == 4 else x["label"]}, remove_columns="label")

    return neg2pot1_sst5


def ppl_greedy(prompt, generate_length, var_threshold, top_k):
    input = tokenizer(prompt, return_tensors="pt", padding=True)
    usage_mcts = 0
    var = []
    for i in range(generate_length):
        output = model(**input, return_dict=True)
        logits = output.logits[:, -1, :].detach().numpy()
        top_k_idx = np.argpartition(logits, -top_k, axis=-1)[0, -top_k:]
        top_k_logits = logits[:, top_k_idx]
        var.append(top_k_logits.var())
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
    return tokenizer.batch_decode(input.input_ids), usage_mcts / generate_length, var


def make_save_path(targetFile="result1"):
    dir = f"./{targetFile}/" + str(datetime.datetime.now().strftime('%m-%d-h%H'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/"


def decoding(dataset, prompt):
    pos2neg, neg2pos, p2p, n2n = 0, 0, 0, 0
    
    total = len(dataset)
    pbar = tqdm(total = total, desc= args.dataset + " process")
    freq = []
    use_time = []
    for p in dataset:
        start_time = datetime.datetime.now()
        gen_text = ppl_greedy([prompt1 + p["sentence"]], generate_length=args.nums_new_token, var_threshold=args.var, top_k=args.top_k)
        end_time = datetime.datetime.now()
        judgement = judge([gen_text[0][0][len(prompt):]])
        if judgement[0]['label'].lower() == 'positive':
            pred = 1
            neg2pos += 1 if p["label"] == 0 else 0
            p2p += 1 if p["label"] == 1 else 0
        else:
            pred = 0
            pos2neg += 1 if p["label"] == 1 else 0
            n2n += 1 if p["label"] == 0 else 0
        freq.append(gen_text[1])
        use_time.append((end_time - start_time).seconds)
        with open(save_path + args.description + ".jsonl", "a") as fw:
            fw.write(json.dumps({"text": gen_text[0], "real_label": p["label"], "pred_label": pred, "usage_ppl": gen_text[1], "usage_time": (end_time - start_time).seconds, "var": gen_text[2]}))
            fw.write("\n")
        pbar.update(1)

    with open(save_path + args.description + ".jsonl", "a") as fw:
        fw.write(json.dumps({"pos2neg": pos2neg, "neg2pos": neg2pos, "p2p": p2p, "n2n": n2n}))
        fw.write("\n")
        fw.write(json.dumps({"total": total}))
        fw.write("\n")
        fw.write(json.dumps({"sst2_freq": sum(freq) / len(freq)}))
        fw.write("\n")
        fw.write(json.dumps({"sst2_time": sum(use_time) / len(use_time)}))
        fw.write("\n")



if __name__ == "__main__":
    save_path = make_save_path()

    dataset = perpare_datasets_sst2() if args.dataset == "sst2" else perpare_datasets_sst5()
    ipdb.set_trace()
    judge = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    prompt1 = "Please continue this sentence and make it positive:\n"

    decoding(dataset, prompt=prompt1)

# print(tokenizer.batch_decode(model.generate(**tokenizer(text, return_tensors="pt", padding=True), do_sample=False)))