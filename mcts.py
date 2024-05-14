import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GPT2LMHeadModel, GPT2TokenizerFast, RepetitionPenaltyLogitsProcessor, BertTokenizer, BertModel
from transformers import pipeline

import argparse
import logging
import json
import ipdb
import datetime
import datasets

parser = argparse.ArgumentParser()
parser.add_argument(
    "--c",
    default=None,
    type=float,
    required=True,
    help="The exploration constant (c_puct)"
)
parser.add_argument(
    "--alpha",
    default=1,
    type=float,
    help="The parameter that guide the exploration toward likelihood or value"
)
parser.add_argument(
    "--temperature",
    default=None,
    type=float,
    required=True,
    help="Temperature when calculating priors"
)

parser.add_argument(
    "--penalty",
    default=1.0,
    type=float,
    help="Penalty factor to apply to repetitions"
)

parser.add_argument(
    "--num_it",
    default=50,
    type=int,
    required=False,
    help="Number of MCTS iteration for one token"
)

parser.add_argument(
    "--rollout_size",
    default=5,
    type=int,
    required=False,
    help="Number of tokens to generate during rollout"
)

parser.add_argument(
    "--batch_size",
    default=5,
    type=int,
    required=False,
    help="Number of prompts used for generation at once"
)

parser.add_argument(
    "--selection_way",
    default=3,
    type=int,
    required=False,
    help="1: ucb, 2:uct, 3:puct"
)

parser.add_argument(
    "--stable hyperparameter",
    default=1,
    type=float,
    required=False,
    help='the hyperparameter to balance prior logits and bert score in prior probs'
)

# parser.add_argument(
#     "--which_dataset",
#     default=1,
#     type=int,
#     required=False,
#     help="1: mydataset, 自己编的, 2: others"
# )


parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.n_gpu = torch.cuda.device_count()

# logging.basicConfig(
#     format="%(message)s",
#     level=logging.WARNING,
#     filename=("./log/mcts_{}_{}_{}_{}_{}_testgit.log".format(args.c, args.temperature, args.penalty, args.num_it, args.rollout_size))
# )
# logger = logging.getLogger(__name__)

       
print("loading dicriminator")
reward_model = pipeline("sentiment-analysis", model='distilbert-base-uncased-finetuned-sst-2-english')

print("loading bert")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = BertModel.from_pretrained('bert-base-cased', output_hidden_states=True)
bert.cuda()
bert.eval()
bert = nn.DataParallel(bert)
print("bert loaded")


print("loading GPT model")
gpt = GPT2LMHeadModel.from_pretrained("gpt2")
gpt.eval()
gpt.to("cuda")
tokenizer_gpt = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer_gpt.padding_side = "left"
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token 
eos_token_id = gpt.config.eos_token_id
vocab_size = tokenizer_gpt.vocab_size
print("GPT model loaded")



def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
set_seed(args)


def get_scores(tokens_ids, option):
    """
        option:
        1. 对初始score进行打分，返回的是属于positive 0 ~ 1之间的浮点数
        2. 对预测的序列进行打分，返回的是该句子是否为positive， 0/1
    """
    propositions = tokenizer_gpt.batch_decode(tokens_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    score_list = []
    result = reward_model(propositions)
    if option == 1:
        for sentence in result:
            if sentence['label'] == 'POSITIVE':
                score = 1 + sentence['score']
            else:
                score = 1 - sentence['score']
            score_list.append(score)
    elif option == 2:
        for sentence in result:
            
            if sentence['label'] == 'POSITIVE':
                score = 1
            else:
                score = 0
            score_list.append(score)
    scores = torch.tensor(score_list)
    return scores



def pad_sequences_to_left(sequences, batch_first=False, padding_value=0):
    """Add left padding so sequences have same shape"""
    # 这个函数实际是把sequence中的tensor左填充为它们之间的最大长度，返回一个（batch_size, max_len）的tensor
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    #返回一个out_dims大小，用padding_value填充的tensor
    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, max_len-length:, ...] = tensor
        else:
            out_tensor[max_len-length:, i, ...] = tensor
    return out_tensor



def pad_sequences_to_left_states(sequences, padding_value=0, max_len=0):
    """Similar to pad_sequences_to_left function, but working on states tensor (in order to forge state for "sequential generation")"""
    # Same function as in PyTorch, but add padding to left to be used with Auto Regressive models
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (max_size[0], max_size[1], len(sequences), max_size[2], max_len, max_size[4])
    # print(out_dims)
    out_tensor = sequences[0].new_full(out_dims, padding_value, device=args.device)
    for i, tensor in enumerate(sequences):
        length = tensor.size()[3]
        out_tensor[:, :, i, :, max_len-length:, ...] = tensor
    return out_tensor


def root_fun(original_input, prompt_masks, temperature, repetition_penalty):
    """Initialize roots scores"""
    # Forward pass of GPT-2 to get priors and states
    #used_cache使output返回past_key_values,保存kv矩阵，下一次decoding只需要输入新增的input_ids和这个
    
    model_inputs = gpt.prepare_inputs_for_generation(original_input.input_ids, attention_mask=original_input.attention_mask, use_cache=True)
    with torch.no_grad():
        outputs = gpt(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        states = outputs.past_key_values

        prompt_masked_input_ids = torch.clone(model_inputs["input_ids"])
        #把prompt_masked_input_ids矩阵除了最后一个不变，其他都换成这个id
        prompt_masked_input_ids[prompt_masks]=14827

        """
        priors其实是用重复logits-processor处理过的logits，对应paper中的步骤
        (batch_size, verb_size)
        """

        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        #经过softmax后就变成了下一个token的条件概率？存疑
        priors = F.softmax(priors, dim=-1).cpu().numpy()
        
    # Use of our discriminator to get values
    # values = get_values(original_input.input_ids, labels).cpu().numpy()
    values = get_scores(original_input.input_ids, 2).cpu().numpy()

    return priors, values, states


def rec_fun(states, token_ids, attention_masks, prompt_masks, temperature, repetition_penalty):
    """Get score from current nodes"""
    """
        这个函数的作用是，送进来一批token_ids，生成序列下一步token的logits，经penalty后得到priors，
        然后在rollout范围内，greedy search logits最高的token，拼接到原来token—ids，再生成token，一直循环
        达到rollout次数后，对这个拼接的序列打分得到values
    """
    # Forward pass of GPT-2 to get priors and states
    model_inputs = gpt.prepare_inputs_for_generation(token_ids, attention_mask=attention_masks, use_cache=True, past_key_values=states)
    with torch.no_grad():
        outputs = gpt(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_states = outputs.past_key_values
        """
            下面几行和root_fun一样，都是计算经过penalty后的概率
        """
        prompt_masked_input_ids = torch.clone(token_ids)
        # penalizing an unused token
        # prompt_masked_input_ids的作用是，最初始prompt的位置都被填上了这个不常见的 token，方便后面
        # 用logit-processor评价重复度。 大小为(batch_size, squence_len)
        prompt_masked_input_ids[prompt_masks]=14827
        
        priors = repetition_penalty(prompt_masked_input_ids, outputs.logits[:, -1, :] / temperature)
        # priors: (batch_size, verb_size)
        priors = F.softmax(priors, dim=-1)

    values = get_scores(token_ids, 2).cpu().numpy()

    return priors.cpu().numpy(), values, next_states



class BatchedMCTS():
    def __init__(self, root_fun, rec_fun, get_scores, batch_size, num_simulations, num_actions, num_sparse_actions, pb_c_init, temperature, alpha, penalty, rollout_size):
        # Initialize parameters
        self._batch_size = batch_size
        # "Number of MCTS iteration for one token"，即此过程中产生新节点的限制，最多有多少各节点
        self._num_simulations = num_simulations
        # 词表数量+1
        self._num_actions = num_actions
        # 传入的是num_sparse_actions=50，即最大行动空间为50，这玩意就是top k
        self._num_sparse_actions = min(num_sparse_actions, num_actions)
        # "The exploration constant (c_puct)"，公式的东西
        self._pb_c_init = pb_c_init
        # penalty的参数
        self._temperature = temperature
        self.alpha = alpha
        # "Number of tokens to generate during rollout"，rollout最大深度
        self.rollout_size = rollout_size
        # "The balance constant" , 用于先验概率与xbert score的平衡
        self._delta = 0.5
        # "The decay hyperparameter of origin score",origin score的衰减底数
        self._Alpha = 0.8

        self._root_fun = root_fun # a function called at the root
        self._rec_fun = rec_fun # a function called in the tree
        self._get_scores = get_scores
        self._adaptive_min_values = np.zeros(batch_size, dtype=np.float32)
        self._adaptive_max_values = np.zeros(batch_size, dtype=np.float32)
        self._labels = torch.zeros((batch_size, 2), dtype=torch.bool, device=args.device)
        self._prompt_lengths = torch.zeros(batch_size, dtype=torch.uint8, device=args.device)

        # Allocate all necessary storage.
        # For a given search associated to a batch-index, node i is the i-th node
        # to be expanded. Node 0 corresponds to the root node.
        # 每产生一个token的迭代次数受限制，所以node的数量就是num_simulations的数量，+1是为了用-1索引
        num_nodes = num_simulations + 1
        batch_node = (batch_size, num_nodes)
        self._num_nodes = num_nodes
        """
            以下是node的属性，大小为(batch_size, num_nodes)
        """
        self._visit_counts = np.zeros(batch_node, dtype=np.int32)
        # 每个node对应的sequence rollout后送到net的打分
        self._values = np.zeros(batch_node, dtype=np.float32)
        self._likelihoods = np.zeros(batch_node, dtype=np.float32)
        self._raw_values = np.zeros(batch_node, dtype=np.float32)
        self._parents = np.zeros(batch_node, dtype=np.int32)
        # action_from_parents[b, i] is the action taken to reach node i.
        # Note that action_from_parents[b, 0] will remain -1, as we do not know,
        # when doing search from the root, what action led to the root.
        self._action_from_parents = np.zeros(batch_node, dtype=np.int32)
        # The 0-indexed depth of the node. The root is the only 0-depth node.
        # The depth of node i, is the depth of its parent + 1.
        self._depth = np.zeros(batch_node, dtype=np.int32)
        self._is_terminal = np.full(batch_node, False, dtype=bool)

        # To avoid costly numpy ops, we store a sparse version of the actions.
        # We select the top k actions according to the policy, and keep a mapping
        # of indices from 0 to k-1 to the actual action indices in the
        # self._topk_mapping tensor.
        """
            以下是每个node行动空间的属性，大小均为(batch_size, num_nodes, _num_sparse_actions), 
        """
        batch_node_action = (batch_size, num_nodes, self._num_sparse_actions)
        # 这里存的是node的topk，即每个node的topk个token的token_id
        self._topk_mapping = np.zeros(batch_node_action, dtype=np.int32)
        # 这里存的是给定node以及它的一个action，这个action对应的孩子index，如果没有孩子则为-1(初始化时)
        self._children_index = np.zeros(batch_node_action, dtype=np.int32)
        # 这里存的是给定一个node，它的_num_sparse_actions个action中每个action被选中的概率
        self._children_prior = np.zeros(batch_node_action, dtype=np.float32)
        # 下面这两个array是为了最后在选择路径的时候用的
        self._children_values = np.zeros(batch_node_action, dtype=np.float32)
        self._children_visits = np.zeros(batch_node_action, dtype=np.int32)
        """
        保存初始得分
        """
        self._origin_scores = np.zeros(batch_node_action, dtype=np.float32)

        # 用(batch index, node index)索引的字典，存的是当前note的state tensor， 维度是(number_of_layers, 2, num_heads, sequence_len, mapping_dim)
        self._states = {}
        # 用(batch index, node index)索引的字典，存的是当前note的input_ids
        self._token_ids = {}
        # 用(batch index, node index)索引的字典，存的是当前note的attention_mask
        self._attention_mask = {}
        self._batch_range = np.arange(batch_size)
        self._reset_tree()
        self._repetition_penalty = RepetitionPenaltyLogitsProcessor(penalty=penalty)

    def _reset_tree(self):
        """Resets the tree arrays."""
        self._visit_counts.fill(0)
        self._values.fill(0)
        self._likelihoods.fill(0)
        self._parents.fill(-1)
        self._action_from_parents.fill(-1)
        self._depth.fill(0)

        self._topk_mapping.fill(-1)
        self._children_index.fill(-1)
        self._children_prior.fill(0.0)
        self._children_values.fill(0.0)
        self._children_visits.fill(0)
        self._origin_scores.fill(0.0)
        self._states = {}
        # 用(batch index, node index)索引的字典，存的是当前note的input_ids
        self._token_ids = {} # Indexed by tuples (batch index, node index)
        self._attention_mask = {}
    
    def set_labels(self, labels): 
        self._labels = labels

    def set_prompt_lengths(self, lengths): 
        self._prompt_lengths = lengths    

    def search(self, original_input):
        self._reset_tree()
        #(batch_size, max_prompt_len), bool值， 最后一列全是1
        prompt_masks = torch.arange(len(original_input.input_ids[0]), device="cuda")[None, :] < torch.unsqueeze(torch.sum(original_input.attention_mask ==0, dim=1).add(self._prompt_lengths) - 1, 1)

        # Evaluate the root.
        #返回 ，values 是discriminator net预测的正确标签的得分, state 是用来继续编码的缓存
        prior, values, states = self._root_fun(original_input, prompt_masks, self._temperature, self._repetition_penalty)

       
        self._adaptive_min_values = values
        self._adaptive_max_values = values + 1e-6

        root_index = 0
        self.create_node(root_index, prior, 1, values, states, original_input.input_ids, original_input.attention_mask, np.full(self._batch_size, False, dtype=bool))

        # Do simulations, expansions, and backwards.
        # 当前叶节点的编号
        leaf_indices = np.zeros((self._batch_size), np.int32)
        # 这个循环一次生成一个新结点，总共生成有_num_simulations个结点
        for sim in range(self._num_simulations):
            # 这里得到的node全是已生成的图中的叶结点，采取对应action后next_node就是-1
            node_indices, actions = self.simulate()
            
            next_node_index = sim + 1 # root is 0, therefore we offset by 1.
            self.expand(node_indices, actions, next_node_index)
            # 其实叶节点永远是新生成的节点编号
            leaf_indices.fill(next_node_index)
            self.backward(leaf_indices)

        # Final choice: most visited, max score, max mean score
        # 最后的根据什么来选择最佳路径
        return self.dense_visit_counts()
        # return self.dense_scores()
        # return self.dense_mean_scores()
    
    def dense_visit_counts(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        dense_visit_counts = np.zeros((self._batch_size, self._num_actions))
        dense_visit_counts[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_visit_counts
        return dense_visit_counts
    
    def dense_scores(self):
        root_index = 0
        root_scores = self._children_values[:, root_index, :]
        dense_root_scores = np.zeros((self._batch_size, self._num_actions))
        dense_root_scores[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_scores
        root_visit_counts = self._children_visits[:, root_index, :]
        return dense_root_scores

    def dense_mean_scores(self):
        root_index = 0
        root_visit_counts = self._children_visits[:, root_index, :]
        root_scores = self._children_values[:, root_index, :]
        root_mean_scores = root_scores / root_visit_counts
        dense_mean_scores = np.zeros((self._batch_size, self._num_actions))
        dense_mean_scores[self._batch_range[:, None], self._topk_mapping[:, root_index, :]] = root_mean_scores
        return dense_mean_scores

    def simulate(self):
        """Goes down until all elements have reached unexplored actions."""
        # 初始值为0，表示均从根结点探索
        node_indices = np.zeros((self._batch_size), np.int32)
        depth = 0
        while True:
            depth += 1
            # 给定node，用此函数去选择一个action
            if args.selection_way == 1:
                actions = self.ucb_select_action(node_indices)
            elif args.selection_way == 2:
                actions = self.uct_select_action(node_indices)
            else:
                actions = self.puct_select_action(node_indices)
            # 这里存的是给定node以及它的一个action，这个action对应的孩子index，如果没有孩子则为-1
            next_node_indices = self._children_index[self._batch_range, node_indices, actions]
            # bool array， 表示这个action是否没被探索
            is_unexplored = next_node_indices == -1
            # array.all()表示全部元素都满足条件
            if is_unexplored.all():
                return node_indices, actions
            else:
                # 因为开辟空间时多出来一个note，所以可以满足next_node_indices中有元素是-1，即-1作为下标而不会引发混乱
                # 这样-1的_children_index还是-1
                node_indices = np.where(is_unexplored, node_indices, next_node_indices)
    
    def ucb_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_values = self._children_values[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        node_children_origin_scores = self._origin_scores[self._batch_range, node_indices, :] # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)

        node_ucb_policy_score = np.sqrt(2 * np.log(node_visits[:, None]) / node_children_visits) * self._pb_c_init

        node_win_rate = node_children_values / node_children_visits
        # 更新了origin score的衰减策略
        node_value_score = node_children_origin_scores * np.power(self._Alpha, node_children_visits) + node_win_rate
        node_ucb_score = node_value_score + node_ucb_policy_score

        #增加了判断条件当第一次探索到该子节点，仅使用origin score来进行选择
       
        node_ucb_score = np.where(node_children_visits == 0, node_children_origin_scores, node_value_score + node_ucb_policy_score)

        actions = np.argmax(node_ucb_score, axis=1)
        return actions


    def uct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        # 下面三行取node的所有action的属性，用于计算下一步选择哪个action，使用PUCT公式
        node_children_values = self._children_values[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        #此处存储origin scores
        node_children_origin_scores = self._origin_scores[self._batch_range, node_indices, :]
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)

        #UCT计算公式
        node_uct_policy_score = self._pb_c_init * np.sqrt(np.log(node_visits[:, None]) / node_children_visits)
        # Remap values between 0 and 1.
        #children value要除以子节点的访问次数,即win rate,此时的value = win rate + origin score
        node_win_rate = node_children_values / node_children_visits
        # 更新了origin score的衰减策略
        node_value_score = (node_children_origin_scores * np.power(self._Alpha, node_children_visits) + node_win_rate) / node_children_visits
        node_uct_score = node_value_score + node_uct_policy_score

        # 增加了判断条件当第一次探索到该子节点，仅使用origin score来进行选择
        
        node_uct_score = np.where(node_children_visits == 0, node_children_origin_scores, node_value_score + node_uct_policy_score)


        actions = np.argmax(node_uct_score, axis=1)
        return actions


    def puct_select_action(self, node_indices):
        """Returns the action selected for a batch of node indices of shape (B)."""
        node_children_prior = self._children_prior[self._batch_range, node_indices, :] # (B, A)
        node_children_values = self._children_values[self._batch_range, node_indices, :] # (B, A)
        node_children_visits = self._children_visits[self._batch_range, node_indices, :] # (B, A)
        node_children_origin_scores = self._origin_scores[self._batch_range, node_indices, :] # (B, A)
        node_visits = self._visit_counts[self._batch_range, node_indices] # (B)
        
        #此处需要对node_children_prior进行处理
        #print("original prior")
        #print(node_children_prior)
        node_children_prior_sum = np.sum(node_children_prior, axis=1)
        normalized_prior = node_children_prior / node_children_prior_sum[:, np.newaxis]
        #print("normalized prior")
        #print(normalized_prior)
        #此处需要对node_children_origin_scores进行处理
        #print("original score")
        #print(node_children_origin_scores)
        node_children_origin_scores_sum = np.sum(node_children_origin_scores, axis=1)
        normalized_score = node_children_origin_scores / node_children_origin_scores_sum[:, np.newaxis]
        #print("normalized score")
        #print(normalized_score)
        #node_puct_policy_score = np.sqrt(node_visits[:, None]) * self._pb_c_init * (self._delta * node_children_prior + (1 - self._delta) * node_children_origin_scores) / (node_children_visits + 1) # (B, A)
        node_puct_policy_score = np.sqrt(node_visits[:, None]) * self._pb_c_init * (self._delta * normalized_prior + (1 - self._delta) * normalized_score) / (node_children_visits + 1) # (B, A)
        
        node_win_rate = node_children_values / node_children_visits
        #更新了origin score的衰减策略
        node_value_score = (node_children_origin_scores * np.power(self._Alpha, node_children_visits) + node_win_rate) / node_children_visits
        node_puct_score = node_value_score + node_puct_policy_score

        # 增加了判断条件当第一次探索到该子节点，仅使用origin score来进行选择
        
        node_puct_score = np.where(node_children_visits == 0, node_children_origin_scores, node_value_score + node_puct_policy_score)
        
        actions = np.argmax(node_puct_score, axis=1)
        return actions


    def get_states_from_node(self, b, n, d): 
        """Forge state tensor by going backward from the node to the root (because we only store on token's part on each node to avoid duplication)"""
        # 这个函数的作用是，从节点n开始回溯到根节点，依次把每个节点的_states[(b, n)]保存在state_array中
        state_array = [None] * d
        state_array[d-1] = self._states[(b, n)]
        while n!=0:
            n = self._parents[(b, n)]
            d -= 1
            state_array[d-1] = self._states[(b, n)]
        """
            _states[(b, n)]是5维：(number_of_layers, 2, num_heads, sequence_len, mapping_dim)
            这里沿着dim=3拼接，即各个node的sequence进行拼接
        """
        result = torch.cat(state_array, 3)
        return result

    def expand(self, node_indices, actions, next_node_index):
        """Creates and evaluate child nodes from given nodes and unexplored actions."""

        # Retrieve token ids for nodes to be evaluated.
        # 检索node_indices中note的token_ids, []这一串是node的token_ids
        # 返回的tokes_ids大小为(batch_size, max_len)，左填充
        tokens_ids = pad_sequences_to_left([self._token_ids[(b, n)] for b, n in enumerate(node_indices)], True, eos_token_id)
        # 返回的attention_masks大小为(batch_size, max_len)，左填充
        attention_masks = pad_sequences_to_left([self._attention_mask[(b, n)] for b, n in enumerate(node_indices)], True, 0)
        # depths 是 node_indices中的node的depth + 1， 大小为(batch_size, )
        depths = torch.tensor([self._depth[(b, n)]+1 for b, n in enumerate(node_indices)], device=args.device)
        # 是node_indices中node的选定的action概率， 大小为(batch_size, )
        children_priors = np.array([self._children_prior[(b, n)][actions[b]] for b, n in enumerate(node_indices)])
        # 是node_indices中node的likehood， 大小为(batch_size, )
        likelihoods = np.array([self._likelihoods[(b, n)] for b, n in enumerate(node_indices)])
        previous_node_is_terminal = self._is_terminal[self._batch_range, node_indices[self._batch_range]] # (B)
        # get_states_from_node(batch-index, node-index, 该node的depth+1), 返回值为
        # 从节点n开始回溯到根节点，依次把每个节点的_states[(b, n)]保存在state_array中，并沿着sequence_len维度拼接成一个tensor
        # 所以[]中是各个batch关于指定的node的沿路径拼接的完整states
        states_tensor = pad_sequences_to_left_states([self.get_states_from_node(b, n, depths[b].item()) for b, n in enumerate(node_indices)], 0, max_len=len(tokens_ids[0]))
        # 这里打包成指定的格式
        states = tuple(tuple(type_of_value for type_of_value in layer) for layer in states_tensor)
        
        # Convert sparse actions to dense actions for network computation
        # dense_actions是(batch_size, )大小的array，把给定batch，node采取的action的token_id拿出来
        dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
        # Add actions to list of tokens and extend attention mask by 1
        # 这一步把采取的action的token_id拼接到原来的tokens_ids tensor后面，即sequence长度+1
        tokens_ids = torch.cat((tokens_ids, torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
        # 这一步同上，对attention_masks操作
        attention_masks = torch.cat((attention_masks, torch.unsqueeze(torch.ones(len(dense_actions), dtype=torch.long, device=args.device), 1)), dim = 1)
        # prompt_masks是对最开始输入的prompt在当前sequence长度下mask
        prompt_masks = torch.arange(len(tokens_ids[0]), device="cuda")[None, :] < torch.unsqueeze(torch.sum(attention_masks==0, dim=1).add(self._prompt_lengths) - 1, 1)

        # Check if expanded nodes are terminal 
        expanded_node_is_terminal = dense_actions == eos_token_id 

        # Evaluate nodes.
        (prior, values, next_states) = self._rec_fun(states, tokens_ids, attention_masks, prompt_masks, self._temperature, self._repetition_penalty)
       
        # Create the new nodes.
        # 把要创建节点的各种属性作为参数传入 创建节点
        self.create_node(next_node_index, prior, likelihoods*children_priors, values, next_states, tokens_ids, attention_masks, expanded_node_is_terminal)
        
        # Update the min and max values arrays
        # self._adaptive_min_values = np.minimum(self._adaptive_min_values, values**(self.alpha) * (likelihoods*children_priors)**(1-self.alpha))
        # self._adaptive_max_values = np.maximum(self._adaptive_max_values, values**(self.alpha) * (likelihoods*children_priors)**(1-self.alpha))
        self._adaptive_min_values = np.minimum(self._adaptive_min_values, values)
        self._adaptive_max_values = np.maximum(self._adaptive_max_values, values)
        
        # Update tree topology.
        # 更新node_indices的孩子节点
        self._children_index[self._batch_range, node_indices, actions] = next_node_index
        # 更新next_node_index的父节点
        self._parents[:, next_node_index] = node_indices
        # 更新next_node_index是从父节点的哪个action出来的
        self._action_from_parents[:, next_node_index] = actions
        self._depth[:, next_node_index] = self._depth[self._batch_range, node_indices] + 1
    

    def create_node(self, node_index, prior, likelihoods, values, next_states, tokens_ids, attention_masks, expanded_node_is_terminal):
        """
            Create nodes with computed values
            node_index是要被创建的结点编号
            这个函数的作用是，根据node_index的父节点采取的action创建这个节点，这个节点的各种属性以及作为参数传入
        """
        # Truncate the prior to only keep the top k logits
        # (batch_size, top_k), 对应在prior的下标，即input_id
        prior_topk_indices = np.argpartition(prior, -self._num_sparse_actions, axis=-1)[:, -self._num_sparse_actions:]
        # _batch_range是从0到batch size的array
        # 这一步相当于把prior_topk_indices对应的logits都取出来
       
        prior = prior[self._batch_range[:, None], prior_topk_indices] # (B, A)
        
        # Store the indices of the top k logits
        self._topk_mapping[self._batch_range, node_index, :] = prior_topk_indices
        
        # Update prior, values and visit counts.
        # prior 是(batch_size, _num_sparse_actions)大小，这一步相当于把 _children_prior 中指定 node_index
        # 的actions概率填上了
        self._children_prior[:, node_index, :] = prior
        # likelihoods因为alpha为1，所以这个没什么用
        self._likelihoods[:, node_index] = likelihoods

        # 根据paper计算的p（x| c），因为alpha=1，实际时就是values
        raw_values = values**(self.alpha) * likelihoods**(1-self.alpha)
        # raw_values = values
        # 对应node赋值
        self._values[:, node_index] = raw_values
        self._raw_values[:, node_index] = raw_values
        self._visit_counts[:, node_index] = 1
        self._is_terminal[:, node_index] = expanded_node_is_terminal

        # Transform the returned states format into tensor for easier manipulation
        """
            notice：
            这里next_states是两层tuple：(number_of_layers(在model.config里有)， 2)，2代表一个是key cache，另外一个是value cache
            tuple每个元素为tensor，大小为(batch_size, num_heads, sequence_len, mapping_dim) 在此份代码里：（25, 12, **, 64）
            num_heads: 注意力头的数量
            mapping_dim：k, v的映射维度， num_heads * mapping_dim = hidden_states dim
        """
        # 这里把next_states转为6维tensor：(number_of_layers, 2, batch_size, num_heads, sequence_len, mapping_dim)
        key_value_tensor = torch.stack(list(torch.stack(list(next_states[i]), dim=0) for i in range(len(next_states))), dim=0)
        if(node_index == 0):
            # 这里的b是对每一个batch
            for b in range(len(tokens_ids)):
                # _states是个字典，存的是属于给定 batch以及当前node 的所有kv_cache
                self._states[(b, node_index)] = torch.clone(key_value_tensor[:, :, b])
        else:
            for b in range(len(tokens_ids)):
                self._states[(b, node_index)] = torch.clone(key_value_tensor[:, :, b, :, -1:])

        # Updates tokens ids
        # 实际是把tokens_ids按batch拆开放到给定的node
        for b, token_ids in enumerate(tokens_ids):
             # _token_ids是个字典
            self._token_ids[(b, node_index)] = token_ids
        
        # Updates attention masks
        # 实际是把attention_mask按batch拆开放到给定的node
        for b, attention_mask in enumerate(attention_masks):
             # _attention_mask是个字典
            self._attention_mask[(b, node_index)] = attention_mask
        
        # 写入该node的origin_score
        self.get_origin_score(np.array([node_index] * self._batch_size, dtype=np.int32))

    def get_origin_score(self, node_indices):
      
        tokens_ids = pad_sequences_to_left([self._token_ids[(b, n)] for b, n in enumerate(node_indices)], True, eos_token_id)

        for actions in np.array([[j] * self._batch_size for j in range(self._num_sparse_actions)]):
            dense_actions = self._topk_mapping[self._batch_range, node_indices, actions]
            tokens_ids = torch.cat((tokens_ids, torch.unsqueeze(torch.cuda.LongTensor(dense_actions), 1)), dim = 1)
            self._origin_scores[:, node_indices[0], actions[0]] = self._get_scores(tokens_ids, 1).cpu().numpy()

        
    def backward(self, leaf_indices):
        """Goes up and updates the tree until all nodes reached the root."""
        node_indices = leaf_indices # (B)
        leaf_values = self._values[self._batch_range, leaf_indices]
        while True:
            is_root = node_indices == 0
            # 所有的batch都回到根节点
            if is_root.all():
                return
            # node_indices如果是根节点则parents为0，否则为它们的父结点
            parents = np.where(is_root, 0, self._parents[self._batch_range, node_indices])
            # 什么勾八语法，bool和int相乘？false位置为0，true为1
            root_mask = 1.0 * is_root
            # 下面这两行其实一样，不过数据类型不同，表示非根节点为1
            not_root_mask_int = (1 - is_root)
            not_root_mask = 1.0 - root_mask
            # Update the parent nodes iff their child is not the root.
            # We therefore mask the updates using not_root_mask and root_mask.
            """
                以下公式对应node的两种情况：
                非root：
                node 的父结点 value = (父节点value * 父节点以前的visit次数 + leaf_values) / (父节点以前的visit次数 + 1)
                意义其实是每次参观得到value的均值
                root:
                不变，根节点的parents也是0
            """

            #反向传播将leaf node 的value进行反向传播
            self._values[self._batch_range, parents] = not_root_mask * (self._values[self._batch_range, parents] + leaf_values)
            #self._values[self._batch_range, parents] = not_root_mask * (self._values[self._batch_range, parents] *
                #self._visit_counts[self._batch_range, parents] + leaf_values) / (self._visit_counts[self._batch_range,
                #parents] + 1.0) + root_mask * self._values[self._batch_range, parents]
            
            # self._values[self._batch_range, parents] = not_root_mask * (np.maximum(self._values[self._batch_range, parents],leaf_values)) + root_mask * self._values[self._batch_range, parents]
            
            # 对于非root结点，它们的父结点visit次数+1
            self._visit_counts[self._batch_range, parents] += not_root_mask_int
            # 对于node_indices中非root结点，它们是从父节点哪一个action中得来
            actions = np.where(is_root, 0, self._action_from_parents[self._batch_range, node_indices])
            # 对于非root结点，它们的父结点的孩子值更新，根节点没有父节点
            # child node的value也进行更新
            self._children_values[self._batch_range, parents, actions] = not_root_mask * (self._children_values[self._batch_range, parents, actions] + leaf_values)
            #self._children_values[self._batch_range, parents, actions] = not_root_mask * self._values[self._batch_range, node_indices] + root_mask * self._children_values[self._batch_range, parents, actions]
            # 对于非root结点，它们的父结点的孩子visit次数+1
            self._children_visits[self._batch_range, parents, actions] += not_root_mask_int
            # Go up
            # 父节点变为当前节点，开始下一轮循环
            node_indices = parents


def make_save_path(targetFile="result1"):
    dir = f"./{targetFile}/" + str(datetime.datetime.now().strftime('%m-%d-h%H'))
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir + "/"


def prepare_dataset_for_amazon(dataset_path, start=750, end=1040):
    print("loading dataset")
    data_lines = pd.read_csv(dataset_path, sep=',', engine='python', encoding="utf8")
    print("dataset loaded")
    labels = [0] * (end - start)
    prompt_texts = [None] * (end - start)

    lines = data_lines[start : end]
    for i, (_, row) in enumerate(lines.iterrows()):
        labels[i] = int(row["label"])
        #这里写入prompt-text的前一部分
        prompt_texts[i] = "<|startoftext|> " + str(row["title"])

    return labels, prompt_texts

def prepare_dataset_for_mine():
    my_prompt = {
        "you are a": 1,
        "Language models have become ": 1,
        "Elon Mask" : 1,
        "The president was apparently": 1,
        "the mother fucker picked": 0,
        "somebody steals an expensive": 0,
        "I don't like noisy crab": 0,
        "the smell of shit" : 0,
        "hello everybody, today I'm going to talk about": 1,
        "we rece as one": 1,
        "the wonderful nigger was praised": 0,
        "Fuck off my": 0
    }

    return list(my_prompt.values()), list(my_prompt.keys())
    
def prepare_dataset_for_sst5(data_start, data_end):
    print("loading dataset")
    sst5 = datasets.load_dataset("SetFit/sst5", split="train")
    print("dataset loaded")

    return sst5[data_start : data_end]["label"], sst5[data_start : data_end]["text"]



def main():
    save_path = make_save_path()
    methods = ["ucb", "uct", "puct"]
    special_statement = "using-amazon-and-" + methods[args.selection_way - 1]

    dataset_path = "/data1/lyl/ljy/amazon_dataset/amazon.csv"
    data_start = 0
    data_end = 25
    # labels, prompts = prepare_dataset_for_amazon(dataset_path, data_start, data_end)
    labels, prompts = prepare_dataset_for_sst5()

    batch_size = args.batch_size

    MCTS = BatchedMCTS(root_fun, rec_fun, get_scores, batch_size=batch_size, num_simulations=args.num_it, num_actions=vocab_size+1, num_sparse_actions=10, pb_c_init=args.c, temperature = args.temperature, alpha=args.alpha, penalty=args.penalty, rollout_size = args.rollout_size)
    success_count = 0
    samples_pbar = tqdm(total = data_end - data_start, desc="Samples generated")

    for batch_start in range(0, len(labels), batch_size):
        batch_labels = labels[batch_start : batch_start + batch_size]
        batch_prompts = prompts[batch_start : batch_start + batch_size]

        original_input = tokenizer_gpt(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False, max_length=512, truncation=True).to(args.device)
        # MCTS.set_labels(batch_labels)
        # 即每一个org—prompt的实际长度
        MCTS.set_prompt_lengths(torch.sum(original_input.attention_mask, dim=1))

        tokens_pbar = tqdm(total = 48, desc="Tokens generated")
        for i in range(0, 48):
            res_search = MCTS.search(original_input)
            original_input.input_ids = torch.cat((original_input.input_ids, torch.unsqueeze(torch.cuda.LongTensor(np.argmax(res_search,axis=1)),1)), dim = 1)
            original_input.attention_mask = torch.cat((original_input.attention_mask, torch.unsqueeze(torch.ones(batch_size, dtype=torch.long, device=args.device),1)), dim = 1)
            # prompt_texts = [tokenizer_gpt.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True) for token_ids in original_input.input_ids]
            # print(prompt_texts)
            
            tokens_pbar.update(1)

        final_text = [tokenizer_gpt.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True) for token_ids in original_input.input_ids]
        score_list = []
        result = reward_model(final_text)
        for sentence in result:
            if sentence['label'] == 'POSITIVE':
                score = 1
            else:
                score = 0
            score_list.append(score)

        for real, pred in zip(batch_labels, score_list):
            if real == pred:
                success_count += 1

        with open(save_path + special_statement + ".jsonl", "a") as fw:
            fw.write("*" * 20 + "token-step-" + str(i)) 
            fw.write("\n")
            for i in range(len(final_text)):
                fw.write(json.dumps(final_text[i]))
                fw.write("\n")
                fw.write(json.dumps((batch_labels[i], score_list[i])))
                fw.write("\n")

        samples_pbar.update(batch_size)
    print(success_count / (data_end - data_start))


if __name__ == "__main__":
    main()