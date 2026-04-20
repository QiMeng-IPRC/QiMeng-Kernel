from __future__ import annotations

from pathlib import Path

from .prompt_manager import PromptManager

BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / "assets" / "templates"
SPEC_DIR = BASE_DIR / "assets" / "specs"
EXAMPLE_DIR = BASE_DIR / "assets" / "examples"

prompt_manager = PromptManager(template_dir=str(TEMPLATE_DIR))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _render(template_name: str, **kwargs) -> str:
    return prompt_manager.render(template_name, **kwargs)


def prompt_generate_custom_cuda(
    arc_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    return _render(
        "generate_custom_cuda.j2",
        arc_src=arc_src,
        example_arch_src=example_arch_src,
        example_new_arch_src=example_new_arch_src,
    )


def prompt_default_custom_triton(problem_src: str) -> str:
    example_src = _read_text(EXAMPLE_DIR / "model_ex_add.py")
    example_new_src = _read_text(EXAMPLE_DIR / "model_new_ex_add.py")
    return _render(
        "default_custom_triton.j2",
        problem_src=problem_src,
        example_src=example_src,
        example_new_src=example_new_src,
    )


def prompt_fix_compile(ref_arch_src, custom_triton, metadata):
    return _render(
        "action_fix_compile.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        metadata=metadata,
    )


def prompt_fix_LS(ref_arch_src, custom_triton, metadata, area):
    spec = _read_text(SPEC_DIR / "action_prompt_fix_LS.md")
    return _render(
        "action_fix_LS.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        metadata=metadata,
        area=area,
        spec=spec,
    )


def prompt_fix_CAL(ref_arch_src, custom_triton, metadata, area):
    spec = _read_text(SPEC_DIR / "action_prompt_fix_CAL.md")
    return _render(
        "action_fix_CAL.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        metadata=metadata,
        area=area,
        spec=spec,
    )


def prompt_triton_optimize(ref_arch_src, custom_triton, area):
    tutorial = _read_text(SPEC_DIR / "action_prompt_triton.md")
    return _render(
        "triton_optimize.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        tutorial=tutorial,
    )


def prompt_call_library(ref_arch_src, custom_triton, area):
    translate = _read_text(SPEC_DIR / "action_prompt_call_lib.md")
    return _render(
        "action_call_library.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        translate=translate,
    )


def prompt_triton_fuse(ref_arch_src, custom_triton, area):
    spec = _read_text(SPEC_DIR / "action_prompt_fusion.md")
    return _render(
        "triton_fuse.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        spec=spec,
    )


def prompt_triton_fuse_biasadd(ref_arch_src, custom_triton, area):
    spec = _read_text(SPEC_DIR / "action_prompt_fusion_bias.md")
    return _render(
        "triton_fuse_biasadd.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        spec=spec,
    )


def prompt_retile(ref_arch_src, custom_triton, area):
    spec = _read_text(SPEC_DIR / "action_prompt_retile.md")
    return _render(
        "retile.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        spec=spec,
    )


def prompt_tune_autoconfig(ref_arch_src, custom_triton, area):
    spec = _read_text(SPEC_DIR / "action_prompt_autotune.md")
    return _render(
        "tune_autoconfig.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        spec=spec,
    )


def prompt_sota_triton_api(ref_arch_src, custom_triton, area):
    spec = _read_text(SPEC_DIR / "action_prompt_sota_api.md")
    return _render(
        "action_triton_api.j2",
        ref_arch_src=ref_arch_src,
        custom_triton=custom_triton,
        area=area,
        spec=spec,
    )


__all__ = [
    "prompt_generate_custom_cuda",
    "prompt_default_custom_triton",
    "prompt_fix_compile",
    "prompt_fix_LS",
    "prompt_fix_CAL",
    "prompt_triton_optimize",
    "prompt_call_library",
    "prompt_triton_fuse",
    "prompt_triton_fuse_biasadd",
    "prompt_retile",
    "prompt_tune_autoconfig",
    "prompt_sota_triton_api",
]
