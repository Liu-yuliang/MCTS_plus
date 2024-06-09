import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="which dataset to evaluate"
    )


    parser.add_argument(
        "--var",
        type=float,
        default=1.0,
        required=False,
        help="To decide the frequency of using ppl decoding strategy"
    )

    parser.add_argument(
        "--nums_new_token",
        type=int,
        default=50,
        required=False,
        help="number of generation length"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        required=False,
        help="top_k"
    )

    parser.add_argument(
        "--description",
        type=str,
        required=True,
        help="special statement"
    )

    # ================= following are ppl hyperparameter ===============

    parser.add_argument(
        "--c",
        default=8,
        type=float,
        # required=True,
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

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()

    return args