import os

import numpy as np
from tqdm import tqdm
import pandas as pd

import torch

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, BertTokenizer, BertModel

import argparse
import logging
from transformers import pipeline

# import ppl_mcts

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

text = ["today is a"]
for i in range(1):
    input = tokenizer(text, return_tensors="pt", padding=True)
    print(input.input_ids.shape)
    output = model(**input, return_dict=True)
    print(output.logits.shape)
    next_text = tokenizer.decode(output.logits)
    print(next_text)
# text = ppl_mcts.mcts_search()
print(text)