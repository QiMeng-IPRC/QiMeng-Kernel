from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from ..agents import KernelEnv, build_agent
from ..kernel_env.KernelEnv import StepStatus
from ..prompt import prompt_default_custom_triton
from ..utils.llm_query_utils import extract_code_block, llm_query
from .args import parser_args


@dataclass
class LaunchResult:
    mode: str
    outputs: list[Path]
    summary: Path | None = None


def _read_problem_source(args) -> tuple[Path, str]:
    path = Path(args.problem_file) if args.problem_file else Path("python/mtmc/prompt/assets/examples/model_ex_add_triton.py")
    return path, path.read_text(encoding="utf-8")


def _query_or_fallback(args, prompt: str, fallback: str) -> str:
    if args.offline:
        return fallback
    response = llm_query(
        prompt,
        model=args.api_model or args.lora_base_model,
        vendor=args.vendor,
        system_prompt=args.system_prompt,
        offline=False,
        max_tokens=4096,
    )
    code = extract_code_block(response)
    return code if code else fallback


def _write_step(out_dir: Path, episode_idx: int, step_idx: int, code: str) -> Path:
    output_path = out_dir / f"episode_{episode_idx}_step_{step_idx}.py"
    output_path.write_text(code, encoding="utf-8")
    return output_path


def _run_episode(args, episode_idx: int, problem_path: Path, problem_src: str) -> list[Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    agent = build_agent(problem_path, problem_src, current_src=problem_src)
    env = KernelEnv(
        agent.problem,
        step_fn=lambda action, prompt, fallback: _query_or_fallback(args, prompt, fallback),
        max_steps=max(args.num_episodes, 1),
    )
    outputs: list[Path] = []

    obs = env.reset(current_src=problem_src)
    baseline_src = problem_src if args.offline else extract_code_block(
        _query_or_fallback(args, prompt_default_custom_triton(problem_src), problem_src)
    ) or problem_src
    outputs.append(_write_step(out_dir, episode_idx, 0, baseline_src))
    env.current_src = baseline_src
    env.problem.current_src = baseline_src
    env.problem.metadata = obs.text

    for step_idx in range(1, max(args.num_episodes, 1)):
        if env.done:
            break
        transition = agent.step(env, out_dir / f"episode_{episode_idx}_step_{step_idx}.py")
        outputs.append(transition.output_path)
        if transition.status in (StepStatus.COMPLETED, StepStatus.STOPPED):
            break

    return outputs


def _inference(args) -> LaunchResult:
    problem_path, problem_src = _read_problem_source(args)
    outputs: list[Path] = []
    for episode_idx in range(max(args.num_episodes, 1)):
        outputs.extend(_run_episode(args, episode_idx, problem_path, problem_src))
    return LaunchResult(mode="inference", outputs=outputs)


def _train(args) -> LaunchResult:
    result = _inference(args)
    summary = Path(args.output_dir) / "train_summary.txt"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(str(p) for p in result.outputs), encoding="utf-8")
    result.summary = summary
    return result


def launch_mtmc(is_train: bool = False):
    args = parser_args(sys.argv[1:], is_train=is_train)
    return _train(args) if is_train or args.run_mode == 0 else _inference(args)


if __name__ == "__main__":
    launch_mtmc()
