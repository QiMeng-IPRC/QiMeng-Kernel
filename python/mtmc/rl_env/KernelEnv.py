from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable


class StepStatus(str, Enum):
    INIT = "init"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    COMPLETED = "completed"


ACTION_TAGS = [
    "STOP",
    "DEBUG_FIX_COMPILE",
    "DEBUG_FIX_LS",
    "DEBUG_FIX_CAL",
    "TRITON_OPTIMIZE",
    "CALL_LIBRARY",
    "FUSION_OPERATION",
    "FUSION_BIAS_ADD",
    "RETILE",
    "AUTOTUNE",
    "SOTA_TRITON_API",
]


ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_TAGS)}
INDEX_TO_ACTION = {idx: name for name, idx in ACTION_TO_INDEX.items()}


@dataclass
class KernelProblem:
    problem_id: str
    ref_arch_path: Path
    problem_src: str
    current_src: str | None = None
    metadata: str | None = None
    area: str = "forward"
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class KernelObservation:
    problem_id: str
    step: int
    status: StepStatus
    ref_arch_path: Path
    problem_src: str
    current_src: str
    metadata: str
    area: str
    last_action: str | None
    last_response: str | None
    available_actions: list[str]
    action_mask: list[int]
    action_map: dict[int, str]
    text: str


@dataclass
class KernelTransition:
    action: str
    prompt: str
    response: str
    code: str
    output_path: Path
    valid: bool
    error: str | None
    reward: float
    done: bool
    status: StepStatus


def _default_available_actions(status: StepStatus, step: int) -> list[str]:
    if status in (StepStatus.STOPPED, StepStatus.COMPLETED):
        return ["STOP"]
    if step == 0:
        return ["TRITON_OPTIMIZE", "CALL_LIBRARY", "FUSION_OPERATION", "FUSION_BIAS_ADD"]
    return ACTION_TAGS[1:]


def _build_mask(available_actions: list[str]) -> list[int]:
    allowed = set(available_actions)
    return [1 if action in allowed else 0 for action in ACTION_TAGS]


def _build_text(obs: "KernelObservation") -> str:
    return (
        f"problem_id={obs.problem_id}\n"
        f"step={obs.step}\n"
        f"status={obs.status.value}\n"
        f"area={obs.area}\n"
        f"last_action={obs.last_action or ''}\n"
        f"metadata={obs.metadata}\n"
        f"available_actions={','.join(obs.available_actions)}\n"
    )


class KernelEnv:
    def __init__(
        self,
        problem: KernelProblem,
        step_fn: Callable[[str, str, str], str] | None = None,
        max_steps: int = 8,
    ):
        self.problem = problem
        self.step_fn = step_fn
        self.max_steps = max_steps
        self.step_count = 0
        self.status = StepStatus.INIT
        self.last_action: str | None = None
        self.last_response: str | None = None
        self.last_error: str | None = None
        self.done = False
        self.current_src = problem.current_src or problem.problem_src

    def reset(self, current_src: str | None = None) -> KernelObservation:
        self.step_count = 0
        self.status = StepStatus.INIT
        self.last_action = None
        self.last_response = None
        self.last_error = None
        self.done = False
        self.current_src = current_src or self.problem.problem_src
        self.problem.current_src = self.current_src
        self.problem.metadata = None
        self.problem.history.clear()
        return self.observe()

    def observe(self) -> KernelObservation:
        metadata = self.problem.metadata or self.last_error or ""
        available_actions = _default_available_actions(self.status, self.step_count)
        obs = KernelObservation(
            problem_id=self.problem.problem_id,
            step=self.step_count,
            status=self.status,
            ref_arch_path=self.problem.ref_arch_path,
            problem_src=self.problem.problem_src,
            current_src=self.current_src,
            metadata=metadata,
            area=self.problem.area,
            last_action=self.last_action,
            last_response=self.last_response,
            available_actions=available_actions,
            action_mask=_build_mask(available_actions),
            action_map=INDEX_TO_ACTION,
            text="",
        )
        obs.text = _build_text(obs)
        return obs

    def validate(self, code: str) -> tuple[bool, str | None]:
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            return False, str(exc)

    def step(self, action: str, prompt: str, output_path: Path, fallback: str) -> KernelTransition:
        if self.done:
            return KernelTransition(
                action="STOP",
                prompt=prompt,
                response=self.last_response or self.current_src,
                code=self.current_src,
                output_path=output_path,
                valid=True,
                error=None,
                reward=0.0,
                done=True,
                status=self.status,
            )

        if action == "STOP":
            self.status = StepStatus.STOPPED
            self.done = True
            self.problem.history.append({"action": action, "output": str(output_path), "status": self.status.value})
            output_path.write_text(self.current_src, encoding="utf-8")
            return KernelTransition(
                action=action,
                prompt=prompt,
                response=self.current_src,
                code=self.current_src,
                output_path=output_path,
                valid=True,
                error=None,
                reward=0.0,
                done=True,
                status=self.status,
            )

        response = self.step_fn(action, prompt, fallback) if self.step_fn else fallback
        code = response.strip() or fallback
        valid, error = self.validate(code)
        reward = 1.0 if valid else -1.0
        if not valid:
            self.status = StepStatus.FAILED
            code = fallback
            reward = -1.0
        else:
            self.status = StepStatus.RUNNING

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
            self.status = StepStatus.COMPLETED if valid else StepStatus.FAILED

        self.current_src = code
        self.problem.current_src = code
        self.last_action = action
        self.last_response = response
        self.last_error = error
        self.problem.metadata = error or ""
        self.problem.history.append(
            {
                "action": action,
                "output": str(output_path),
                "status": self.status.value,
                "valid": str(valid),
            }
        )
        output_path.write_text(code, encoding="utf-8")
        return KernelTransition(
            action=action,
            prompt=prompt,
            response=response,
            code=code,
            output_path=output_path,
            valid=valid,
            error=error,
            reward=reward,
            done=self.done,
            status=self.status,
        )
