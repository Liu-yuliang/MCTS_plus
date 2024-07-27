import torch
import torch.nn.functional as F
import numpy as np
import os
from tqdm import tqdm
import pandas as pd

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RepetitionPenaltyLogitsProcessor, BertTokenizer, BertModel
from transformers import pipeline
import ipdb
from mcts_node import Node

class Tree:
    def __init__(self, num_rollout, rollout_len, vocabulary_size, num_branch, num_node, pb_c_init, temperature, alpha, penalty):
        self._nodes = []
        self._num_node = num_node
        self._num_branch = num_branch
        self._vocabulary_size = vocabulary_size
        
        self._num_rollout = num_rollout
        self._rollout_len = rollout_len

        self._model = GPT2LMHeadModel.from_pretrained("gpt2")
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        
          
    def search(self, prompt):
        model_inputs = self._tokenizer(prompt)
        outputs = self._model(**model_inputs, return_dict=True)
        logits = outputs.logits

        pass
        
        
    def create_root(self):
        
        root = Node()

        pass

    
    def get_priors(self, logits, temperature):
        # prompt_masked_input_ids = torch.clone(model_inputs["input_ids"])
        # prompt_masked_input_ids[prompt_masks]=14827
        # priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        # priors = F.softmax(priors, dim=-1).cpu().numpy()
        pass

    def recusive_select(self):
        pass


    def creat_new_node(self, node_index, father_index, prior, values, token_id):
        node = Node(node_index, father_index, priors, value, token_id)
        pass


    def rollout_out(self):
        pass


    def backward(self):
        pass


        

