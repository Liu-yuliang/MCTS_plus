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
    def __init__(self, num_rollout, rollout_len, num_branch, num_node, pb_c_init, rollout_gen_config, device, temperature, penalty):
        self._nodes = []
        self._num_node = num_node
        self._num_branch = num_branch
        
        self._num_rollout = num_rollout
        self._rollout_len = rollout_len
        self._rollout_gen_config = rollout_gen_config

        self._model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        self._tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._device = device

        self._repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty)
        self._temperature=temperature
        self._pb_c_init = pb_c_init
          
    def search(self, prompt):
        # ipdb.set_trace()
        self.create_root(prompt)

        for node_idx in range(1, self._num_node + 1):
            # selection
            leaf_node, selected_branch, sequence_tokens = self.recusive_select()
            sequence_tokens = torch.cat((torch.tensor(sequence_tokens), torch.tensor([leaf_node._branch_token_ids[selected_branch]])))
            
            # rollout to get the value for expanding a new node
            value = self.rollout_out(sequence_tokens)

            # creating a new node
            new_node = self.creat_new_node(
                node_idx=node_idx,
                father_node=leaf_node,
                selected_branch=selected_branch,
                sequence_tokens=sequence_tokens,
                value=value
            )
            self._nodes.append(new_node)

            # backpropagation
            self.backward(new_node)

        # select the sequence path with most visiting count
        sequence = self.get_path()

        return self._tokenizer.decode(sequence)

        
    def create_root(self, prompt: str):
        model_inputs = self._tokenizer(prompt, return_tensors="pt")
        outputs = self._model(**{k: v.to(self._device) for k, v in model_inputs.items()}, return_dict=True)
        logits = outputs.logits
        root_value = self.get_value(torch.squeeze(model_inputs.input_ids))
        all_priors = self.get_priors(torch.squeeze(model_inputs.input_ids))
        topk_token_ids = np.argpartition(all_priors, -self._num_branch, axis=-1)[-self._num_branch:]
        priors = all_priors[topk_token_ids]

        root = Node(
            idx = 0,
            father_idx=-1,
            from_father_which_branch=-1,
            token_id=torch.squeeze(model_inputs.input_ids).detach().numpy(),
            value=root_value,
            branch_prior=priors,
            branch_token_ids=topk_token_ids,
            num_branch=self._num_branch,
            pb_c_init=self._pb_c_init
        )
        self._nodes.append(root)

    
    def get_priors(self, sequence_tokens: torch.Tensor):
        inputs = {"input_ids": sequence_tokens.to(self._device)}
        outputs = self._model(**inputs, return_dict=True)
        prompt_masked_input_ids = torch.clone(inputs["input_ids"])
        prompt_masked_input_ids[:-1]=14827
        prompt_masked_input_ids = torch.unsqueeze(prompt_masked_input_ids, dim=0)
        priors = self._repetition_penalty(prompt_masked_input_ids, torch.unsqueeze(outputs.logits[-1, :], dim=0) / self._temperature)
        priors = F.softmax(priors, dim=-1).detach().cpu().numpy()
        priors = np.squeeze(priors)
        return priors


    def recusive_select(self):
        dynamic_node = self._nodes[0]
        branch_puct_score = dynamic_node.get_puct_score()
        selected_branch = np.argmax(branch_puct_score)
        input_ids = torch.tensor(dynamic_node._token_id)

        while selected_branch in dynamic_node._children_branch:
            dynamic_node = self._nodes[dynamic_node._children_idx[selected_branch]]
            branch_puct_score = dynamic_node.get_puct_score()
            selected_branch = np.argmax(branch_puct_score)
            input_ids = torch.cat((input_ids, torch.tensor([dynamic_node._token_id])))
        return dynamic_node, selected_branch, input_ids


    def creat_new_node(self, node_idx, father_node, selected_branch, sequence_tokens, value):
        father_node._children_branch.append(selected_branch)
        father_node._children_idx.update({selected_branch: node_idx})

        all_priors = self.get_priors(sequence_tokens)
        topk_token_ids = np.argpartition(all_priors, -self._num_branch, axis=-1)[-self._num_branch:]
        priors = all_priors[topk_token_ids]

        new_node = Node(
            idx=node_idx,
            father_idx=father_node._idx,
            from_father_which_branch=selected_branch,
            token_id=father_node._branch_token_ids[selected_branch],
            value=value,
            num_branch=self._num_branch,
            branch_prior=priors,
            branch_token_ids=topk_token_ids,
            pb_c_init=self._pb_c_init
            )
        
        return new_node


    def rollout_out(self, input_ids: torch.Tensor):
        values = []
        input_ids = torch.unsqueeze(input_ids, dim=0)
        input = {"input_ids": input_ids.to(self._device), "attention_mask": torch.ones((1, input_ids.shape[0])).to(self._device)}
        for i in range(self._num_rollout):
            outputs = self._model.generate(**input, generation_config=self._rollout_gen_config)
            values.append(self.get_value(outputs))
        return sum(values) / len(values)


    def get_value(self, input_ids: torch.Tensor):
        return np.random.randn()


    def backward(self, node):
        aggregate_value = node._value        
        while node._father_idx != -1:
            node._value = (node._value * node._visit_counts + aggregate_value) / (node._visit_counts + 1)
            node._visit_counts += 1
            father_node = self._nodes[node._father_idx]
            father_node._branch_value[node._from_father_which_branch] = node._value
            father_node._branch_visit[node._from_father_which_branch] += 1
            node = father_node


    def get_path(self):
        node = self._nodes[0]
        path = node._token_id
        while node._children_branch != []:
            selected_branch = np.argmax(node._branch_visit)
            children_idx = node._children_idx[selected_branch]
            node = self._nodes[children_idx]
            path = np.append(path, node._token_id)
        return torch.tensor(path)


