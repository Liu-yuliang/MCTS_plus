import numpy as np
import torch


class Node:
    def __init__(self, idx, father_idx, branch_prior, value, token_id):
        self._idx = idx
        self._father_idx = father_idx
        self._children_idx = []
        self._branch_prior = branch_prior
        self._token_id = token_id
        self._value = value
        self._visit_counts

