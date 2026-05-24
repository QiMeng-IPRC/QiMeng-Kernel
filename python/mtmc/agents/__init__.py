from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from ..kernel_env.KernelEnv import (
    ACTION_TAGS,
    KernelEnv,
    KernelObservation,
    KernelProblem,
    KernelTransition,
    StepStatus,
)
from ..prompt import (
    prompt_call_library,
    prompt_default_custom_triton,
    prompt_fix_CAL,
    prompt_fix_LS,
    prompt_fix_compile,
    prompt_retile,
    prompt_sota_triton_api,
    prompt_triton_fuse,
    prompt_triton_fuse_biasadd,
    prompt_triton_optimize,
    prompt_tune_autoconfig,
)


class ActionPolicy(Protocol):
    def __call__(self, obs: KernelObservation) -> str: ...


def build_prompt(action: str, obs: KernelObservation) -> str:
    ref_src = obs.problem_src
    cur_src = obs.current_src
    metadata = obs.text
    area = obs.area
    if action == "DEBUG_FIX_COMPILE":
        return prompt_fix_compile(ref_src, cur_src, metadata)
    if action == "DEBUG_FIX_LS":
        return prompt_fix_LS(ref_src, cur_src, metadata, area)
    if action == "DEBUG_FIX_CAL":
        return prompt_fix_CAL(ref_src, cur_src, metadata, area)
    if action == "TRITON_OPTIMIZE":
        return prompt_triton_optimize(ref_src, cur_src, area)
    if action == "CALL_LIBRARY":
        return prompt_call_library(ref_src, cur_src, area)
    if action == "FUSION_OPERATION":
        return prompt_triton_fuse(ref_src, cur_src, area)
    if action == "FUSION_BIAS_ADD":
        return prompt_triton_fuse_biasadd(ref_src, cur_src, area)
    if action == "RETILE":
        return prompt_retile(ref_src, cur_src, area)
    if action == "AUTOTUNE":
        return prompt_tune_autoconfig(ref_src, cur_src, area)
    if action == "SOTA_TRITON_API":
        return prompt_sota_triton_api(ref_src, cur_src, area)
    return prompt_default_custom_triton(ref_src)


def heuristic_policy(obs: KernelObservation) -> str:
    if obs.status in (StepStatus.STOPPED, StepStatus.COMPLETED):
        return "STOP"
    if obs.step == 0:
        return "TRITON_OPTIMIZE"
    if obs.last_error:
        return "DEBUG_FIX_COMPILE"
    if "CALL_LIBRARY" in obs.available_actions and obs.step < 2:
        return "CALL_LIBRARY"
    return next((action for action in obs.available_actions if action != "STOP"), "STOP")


@dataclass
class AgentStep:
    action: str
    prompt: str
    result: KernelTransition


@dataclass
class MTMCAgent:
    problem: KernelProblem
    policy: ActionPolicy = heuristic_policy
    steps: list[AgentStep] = field(default_factory=list)

    def decide(self, obs: KernelObservation) -> str:
        return self.policy(obs)

    def step(self, env: KernelEnv, output_path: Path) -> KernelTransition:
        obs = env.observe()
        action = self.decide(obs)
        prompt = build_prompt(action, obs)
        fallback = obs.current_src or obs.problem_src
        result = env.step(action, prompt, output_path, fallback)
        self.steps.append(AgentStep(action=action, prompt=prompt, result=result))
        return result


def build_agent(problem_path: Path, problem_src: str, current_src: str | None = None, area: str = "forward") -> MTMCAgent:
    problem = KernelProblem(
        problem_id=problem_path.stem,
        ref_arch_path=problem_path,
        problem_src=problem_src,
        current_src=current_src or problem_src,
        area=area,
    )
    return MTMCAgent(problem=problem)


__all__ = [
    "ACTION_TAGS",
    "ActionPolicy",
    "AgentStep",
    "KernelEnv",
    "KernelObservation",
    "KernelProblem",
    "KernelTransition",
    "MTMCAgent",
    "build_agent",
    "build_prompt",
    "heuristic_policy",
]
