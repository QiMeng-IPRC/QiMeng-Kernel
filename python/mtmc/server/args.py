from __future__ import annotations

import argparse
from typing import List


def parser_args(args: List[str], is_train: bool = False):
    parser = argparse.ArgumentParser(description="MTMC Arguments")

    # Frozen LM
    parser.add_argument(
        "--lora-base-model",
        "--model",
        type=str,
        default="models/Llama-3.2-1B",
        help="The frozen base language model.",
    )
    parser.add_argument(
        "--token-normalize-mode",
        type="string",
    )

    parser.add_argument(
        "--seed", type=int, default=77, help="the seed of the experiment."
    )

    parser.add_argument(
        "--offline",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        help="Online or offline training, default is False, which means online training.",
    )

    parser.add_argument(
        "--load_8bit",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=False,
        help="load 8-bit quantized model.",
    )
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="sum",
        choices=["token", "word", "sum"],
        help="the normalize mode of the lora model.",
    )

    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1048575,
        help="the number of parallel game environments.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="the number of parallel game environments.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=4,
        help="the number of steps to run in each environment per policy rollout.",
    )
    parser.add_argument(
        "--num_critic_warmup_steps",
        type=int,
        default=512,
        help="the number of steps to warm up critic.",
    )
    parser.add_argument(
        "--gradient_checkpointing_steps",
        type=int,
        default=8,
        help="the number of steps for gradient checkpointing",
    )
    parser.add_argument(
        "--len_epoch",
        type=int,
        default=1024,
        help="the length of steps of each epoch, only for save checkpoint.",
    )

    parser.add_argument(
        "--policy_learning_rate",
        type=float,
        default=1e-6,
        help="the learning rate of the optimizer.",
    )
    parser.add_argument(
        "--value_learning_rate",
        type=float,
        default=5e-5,
        help="the learning rate of the optimizer.",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--norm_adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--update_epochs", type=int, default=1, help="the K epochs to update the policy"
    )
    parser.add_argument(
        "--policy_num_minibatches",
        type=int,
        default=32,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--value_num_minibatches",
        type=int,
        default=4,
        help="the number of mini-batches",
    )
    parser.add_argument(
        "--clip_vloss",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.",
    )
    parser.add_argument(
        "--clip_coef",
        type=float,
        default=0.2,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent_coef", type=float, default=0.01, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--vf_coef", type=float, default=0.5, help="coefficient of the value function"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=0.02,
        help="the target KL divergence threshold",
    )

    # Env params
    parser.add_argument(
        "--dataset",
        type=str,
        default="kernelbench",
        choices=["kernelbench", "kernelbenchplus", "TritonBench_T_Model"],
        help="the path of dataset, 'kernelbenchplus' for training, and 'kernelbench' for evaluating.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=["level1", "level2", "level3"],
        help="the levels of dataset.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default="random",
        help="the method of sampling from dataset.",
        choices=["random", "seq"],
    )
    parser.add_argument("--vendor", type=str, help="the LLM API vendor.")
    parser.add_argument("--api_model", type=str, help="the LLM API model.")

    parser.add_argument(
        "--debug",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        help="whehter print the debug information.",
    )
    parser.add_argument(
        "--resume",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        help="whehter resume from previous checkpoint.",
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="the path of checkpoint needs to be resumed.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="checkpoints",
        help="the path to save the checkpoint.",
    )
    parser.add_argument(
        "--record_path",
        type=str,
        default="runs",
        help="the path to save the tensorboard results.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="the path of dir of checkpoint.",
    )
    parser.add_argument(
        "--distributed",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        help="whehter use distributed training.",
    )

    args = parser.parse_args()
    if args.run_mode == 1 or args.run_mode == 2:
        args.offline = False
    args.batch_size = int(args.num_envs * args.num_steps)
    args.policy_minibatch_size = max(
        int(args.batch_size // args.policy_num_minibatches), 1
    )
    args.value_minibatch_size = max(
        int(args.batch_size // args.value_num_minibatches), 1
    )

    return args
