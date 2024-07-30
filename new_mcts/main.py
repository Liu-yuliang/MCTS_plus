import argparse
import torch
import mcts_tree
from transformers import GenerationConfig

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_rollout",
        default=5,
        type=int,
        help="How many times repeat for rollout"
    )
    parser.add_argument(
        "--rollout_len",
        default=20,
        type=int,
        help="How many tokens are generated in rollout"
    )
    parser.add_argument(
        "--num_branch",
        default=10,
        type=int,
        help="Number of branches in a single node"
    )
    parser.add_argument(
        "--num_node",
        default=30,
        type=int,
        help="Number of nodes in a mcts tree"
    )
    parser.add_argument(
        "--c",
        default=8,
        type=float,
        # required=True,
        help="The exploration constant (c_puct)"
    )

    parser.add_argument(
        "--temperature",
        default=1.2,
        type=float,
        # required=True,
        help="Temperature when calculating priors"
    )

    parser.add_argument(
        "--penalty",
        default=1.0,
        type=float,
        help="Penalty factor to apply to repetitions"
    )

    args = parser.parse_args()

    return args

def main():
    args = get_args()
    prompt = "The US president was attacked during his speech about economy and employment, and"
    device = torch.device("cuda:3")
    rollout_gen_config = GenerationConfig(
        max_new_tokens=args.rollout_len,
        min_new_tokens=args.rollout_len,
        do_sample=True,
        top_k=20
    )

    tree = mcts_tree.Tree(
        num_rollout=args.num_rollout,
        rollout_len=args.rollout_len,
        num_branch=args.num_branch,
        num_node=args.num_node,
        pb_c_init=args.c,
        rollout_gen_config=rollout_gen_config,
        device=device,
        temperature=args.temperature,
        penalty=args.penalty
    )

    print(tree.search(prompt))



if __name__ == "__main__":
    main()