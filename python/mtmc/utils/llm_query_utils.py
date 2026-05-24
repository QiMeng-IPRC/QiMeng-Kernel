from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def build_model_name(vendor: str | None, model: str) -> str:
    if vendor and "/" not in model:
        return f"{vendor}/{model}"
    return model


def llm_query(
    prompt: str | list[dict[str, str]],
    *,
    model: str,
    vendor: str | None = None,
    system_prompt: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    offline: bool = False,
    **kwargs: Any,
) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        messages.extend(prompt)

    if offline:
        return prompt if isinstance(prompt, str) else "\n".join(message.get("content", "") for message in messages)

    try:
        import litellm
    except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("litellm is required for online inference") from exc

    response = litellm.completion(
        model=build_model_name(vendor, model),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content or ""


def extract_code_block(text: str) -> str:
    start = text.find("```")
    if start < 0:
        return text.strip()
    end = text.find("```", start + 3)
    if end < 0:
        return text[start + 3 :].strip()
    block = text[start + 3 : end].strip()
    if block.startswith("python\n"):
        return block[7:].strip()
    return block


def llm_query_batch(
    prompts: Iterable[str],
    *,
    model: str,
    vendor: str | None = None,
    **kwargs: Any,
) -> list[str]:
    return [llm_query(prompt, model=model, vendor=vendor, **kwargs) for prompt in prompts]
