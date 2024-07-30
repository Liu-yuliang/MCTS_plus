import numpy as np
import torch
import math

class Node:
    def __init__(self, idx, father_idx, from_father_which_branch, token_id, value, branch_prior, branch_token_ids, num_branch, pb_c_init):
        self._idx = idx
        self._father_idx = father_idx
        self._from_father_which_branch = from_father_which_branch
        
        self._token_id = token_id
        self._value = value
        self._visit_counts = 0

        self._branch_prior = branch_prior
        self._branch_value = np.zeros(num_branch)
        self._branch_visit = np.zeros(num_branch, dtype=np.int32)
        self._branch_token_ids = branch_token_ids
        
        self._children_branch = []
        self._children_idx = {}     # mapping from branch order to children node index in node list
        self._pb_c_init = pb_c_init


    def get_puct_score(self):
        value_score = self._branch_value 
        if self._visit_counts == 0:
            policy_score = self._branch_prior
        else:
            policy_score = self._branch_prior * np.sqrt(self._visit_counts) / (1 + self._branch_visit)
        return value_score + self._pb_c_init * policy_score


