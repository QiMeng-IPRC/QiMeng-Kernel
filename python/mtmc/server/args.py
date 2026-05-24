from __future__ import annotations

import argparse
from typing import List


def _add_bool_arg(
    parser: argparse.ArgumentParser, name: str, *, default: bool = False, **kwargs
) -> None:
    parser.add_argument(
        name, action=argparse.BooleanOptionalAction, default=default, **kwargs
    )


def parser_args(args: List[str], is_train: bool = False):
    parser = argparse.ArgumentParser(description="MTMC Arguments")

    parser.add_argument(
        "--lora-base-model",
        "--model",
        dest="lora_base_model",
        type=str,
        default="models/Llama-3.2-1B",
        help="The frozen base language model.",
    )
    parser.add_argument(
        "--token-normalize-mode",
        type=str,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=77, help="The seed.")
    _add_bool_arg(parser, "--offline", help="Run in offline mode.")
    _add_bool_arg(parser, "--load_8bit", help="Load 8-bit model.")
    parser.add_argument(
        "--normalize_mode",
        type=str,
        default="sum",
        choices=["token", "word", "sum"],
        help="The normalize mode of the lora model.",
    )

    parser.add_argument("--total_timesteps", type=int, default=1048575)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--num_critic_warmup_steps", type=int, default=512)
    parser.add_argument("--gradient_checkpointing_steps", type=int, default=8)
    parser.add_argument("--len_epoch", type=int, default=1024)

    parser.add_argument("--policy_learning_rate", type=float, default=1e-6)
    parser.add_argument("--value_learning_rate", type=float, default=5e-5)
    _add_bool_arg(parser, "--anneal-lr", default=True)
    _add_bool_arg(parser, "--norm_adv", default=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--update_epochs", type=int, default=1)
    parser.add_argument("--policy_num_minibatches", type=int, default=32)
    parser.add_argument("--value_num_minibatches", type=int, default=4)
    _add_bool_arg(parser, "--clip_vloss", default=True)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--target_kl", type=float, default=0.02)

    parser.add_argument(
        "--dataset",
        type=str,
        default="kernelbench",
        choices=["kernelbench", "kernelbenchplus", "TritonBench_T_Model"],
    )
    parser.add_argument("--levels", nargs="+", default=["level1", "level2", "level3"])
    parser.add_argument(
        "--sample_mode", type=str, default="random", choices=["random", "seq"]
    )
    parser.add_argument("--vendor", type=str, default=None)
    parser.add_argument("--api_model", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--problem_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument(
        "--action_policy", type=str, default="heuristic", choices=["heuristic", "llm"]
    )
    parser.add_argument(
        "--action_sequence",
        nargs="+",
        default=[
            "TRITON_OPTIMIZE",
            "CALL_LIBRARY",
            "FUSION_OPERATION",
            "FUSION_BIAS_ADD",
            "RETILE",
            "AUTOTUNE",
            "SOTA_TRITON_API",
        ],
    )

    _add_bool_arg(parser, "--debug", help="Print debug information.")
    _add_bool_arg(parser, "--resume", help="Resume from checkpoint.")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="checkpoints")
    parser.add_argument("--record_path", type=str, default="runs")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    _add_bool_arg(parser, "--distributed", help="Use distributed training.")

    parser.add_argument(
        "--run_mode",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0=train, 1=inference, 2=evaluate.",
    )

    parsed = parser.parse_args(args)
    if parsed.run_mode in (1, 2):
        parsed.offline = False

    parsed.batch_size = int(parsed.num_envs * parsed.num_steps)
    parsed.policy_minibatch_size = max(
        int(parsed.batch_size // parsed.policy_num_minibatches), 1
    )
    parsed.value_minibatch_size = max(
        int(parsed.batch_size // parsed.value_num_minibatches), 1
    )
    parsed.is_train = is_train
    return parsed
