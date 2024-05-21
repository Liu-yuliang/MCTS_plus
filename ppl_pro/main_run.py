import os

import numpy as np
from tqdm import tqdm
import pandas as pd

import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertModel

import argparse
import logging
from transformers import pipeline

import ppl_mcts

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

text = ["the wonderful nigger was praised"]

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




# print(tokenizer.batch_decode(model.generate(**tokenizer(text, return_tensors="pt", padding=True), do_sample=False)))