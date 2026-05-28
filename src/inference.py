"""
Inference script for PaintBench.

Feeds benchmark input images + instructions to a model and saves the output images.
Logs comprehensive metrics: GPU hardware, timing, memory usage, throughput, etc.
No evaluation or scoring is performed — only model outputs are saved.

Usage:
    python src/inference.py --model flux2-dev          --benchmark benchmarks/PaintBench --out-dir model_outputs
    python src/inference.py --model qwen-image-edit    --benchmark benchmarks/PaintBench --out-dir model_outputs
    python src/inference.py --model instruct-pix2pix   --benchmark benchmarks/PaintBench --out-dir model_outputs
    python src/inference.py --model longcat-image-edit --benchmark benchmarks/PaintBench --out-dir model_outputs
    python src/inference.py --model bagel              --benchmark benchmarks/PaintBench --out-dir model_outputs
    python src/inference.py --model nano-banana-2      --benchmark benchmarks/PaintBench --out-dir model_outputs
    python src/inference.py --model gpt-image-2        --benchmark benchmarks/PaintBench --out-dir model_outputs

    # Specific tasks only
    python src/inference.py --model flux2-dev --benchmark benchmarks/PaintBench --out-dir model_outputs --tasks translation,rotation

    # Limit problems per task (useful for quick sanity checks)
    python src/inference.py --model flux2-dev --benchmark benchmarks/PaintBench --out-dir model_outputs --max-problems 5
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

from benchmark_source import BenchmarkSource, Problem, parse_benchmark_arg

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─── .env loader ──────────────────────────────────────────────────────────────
# Auto-load a repo-root ``.env`` (gitignored) before any model classes look at
# the environment. Inline-parses to preserve values containing shell-special
# characters like ``|`` or ``$`` that ``set -a; source`` would mishandle.
# Existing environment variables take precedence over the file (CLI overrides).

def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env_file(Path(__file__).resolve().parent.parent / ".env")


# ─── Transient-error predicate ────────────────────────────────────────────────
# Markers for transient/retryable errors hit during long benchmark runs through
# any HTTP gateway: rate limits + transport-layer blips. Extracted to module
# scope so it's testable without spinning up a model client.

_TRANSIENT_ERROR_MARKERS = (
    "429", "502", "503", "504",
    "Server disconnected",
    "ConnectionError", "RemoteProtocolError", "RemoteDisconnected",
    "ReadTimeout", "ConnectTimeout",
)


def _is_transient_error(exc: BaseException) -> bool:
    """True if ``exc``'s string contains a known transient marker.

    Matches HTTP 429/5xx and transport-layer disconnects/timeouts. Long
    multi-hundred-call API runs through any HTTP gateway will hit a few of
    these; without retry, each one orphans a problem.
    """
    return any(m in str(exc) for m in _TRANSIENT_ERROR_MARKERS)


# ─── Models ───────────────────────────────────────────────────────────────────
# Each model class implements:
#   load_model()              — download / initialise weights
#   generate(image, prompt)   — returns one of:
#                                 - PIL Image,
#                                 - (PIL Image, reasoning_text), or
#                                 - (PIL Image, reasoning_text, extra_sidecars)
#                               where extra_sidecars maps a filename-tail
#                               (e.g. "_trace.json") to bytes the orchestrator
#                               should write next to the output PNG. See
#                               _unpack_generate_result.
# Register new models in _REGISTRY at the bottom of this section.

ModelOutput = Union[
    Image.Image,
    Tuple[Image.Image, str],
    Tuple[Image.Image, str, Dict[str, bytes]],
]


_TRACKED_PIPELINE_PARAMS = {
    "num_inference_steps", "guidance_scale", "image_guidance_scale",
    "strength", "true_cfg_scale", "negative_prompt", "num_images_per_prompt",
}


class BaseModel:
    """Base class for inference models.

    Thread-safety contract: ``generate()`` is invoked from a
    ``ThreadPoolExecutor`` whenever ``--workers N > 1`` is passed. Wrappers
    that share mutable per-pipeline state across calls (the diffusers
    ``self.pipe`` pattern, where the scheduler holds a mutable
    ``_step_index`` between sampling steps) must serialize ``generate()``
    against itself, otherwise two concurrent calls race on the scheduler
    and one thread eventually reads past ``sigmas[-1]`` with an
    ``IndexError: index N+1 is out of bounds for dimension 0 with size N+1``
    from inside ``scheduling_flow_match_euler_discrete.step()``. The
    failure is non-deterministic — most calls succeed, one in every few
    hundred to thousand drops out.

    Convention for shared-pipeline wrappers: hold a ``self._lock =
    threading.Lock()`` (initialised in ``__init__``) and wrap the body of
    ``generate()`` in ``with self._lock:``. ``WORKERS > 1`` then only buys
    a small CPU/GPU overlap on pre/post-processing (the GPU step itself
    is still the bottleneck), but it never crashes.

    Wrappers without shared mutable state (subprocess-per-call CLI
    agents) or with their own async concurrency surface
    (``asyncio.Semaphore``-backed API clients) don't need the lock.
    """

    def load_model(self) -> None:
        raise NotImplementedError

    def generate(self, image: Image.Image, instruction: str) -> ModelOutput:
        raise NotImplementedError

    def output_dir_slug(self, registry_key: str) -> str:
        """Return the subdir name under ``--out-dir`` for this model's
        outputs. Defaults to the registry key. Subclasses may override
        to encode model-variant info so different backends don't clobber
        each other's outputs under the same registry entry."""
        return registry_key

    def get_model_info(self) -> Dict:
        """Return model metadata and effective pipeline parameters for logging."""
        pipe = getattr(self, "pipe", None)
        pipeline_defaults = {}
        if pipe is not None:
            sig = inspect.signature(pipe.__call__)
            pipeline_defaults = {
                name: param.default
                for name, param in sig.parameters.items()
                if name in _TRACKED_PIPELINE_PARAMS
                and param.default is not inspect.Parameter.empty
            }
        overrides = getattr(self, "pipeline_kwargs", {})
        return {
            "model_id": getattr(self, "MODEL_ID", None),
            "pipeline_class": type(pipe).__name__ if pipe is not None else None,
            "device": getattr(self, "device", None),
            "torch_dtype": str(getattr(self, "dtype", None)),
            "seed": getattr(self, "seed", None),
            "pipeline_params": {**pipeline_defaults, **overrides},
        }


class InstructPix2PixModel(BaseModel):
    """timbrooks/instruct-pix2pix via diffusers StableDiffusionInstructPix2PixPipeline."""

    MODEL_ID = "timbrooks/instruct-pix2pix"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str      = "float16",
        seed: int             = 42,
        num_inference_steps: Optional[int]   = None,
        guidance_scale: Optional[float]      = None,
        image_guidance_scale: Optional[float]= None,
        **_,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = getattr(torch, torch_dtype)
        self.seed   = seed
        self.pipeline_kwargs = {k: v for k, v in {
            "num_inference_steps":  num_inference_steps,
            "guidance_scale":       guidance_scale,
            "image_guidance_scale": image_guidance_scale,
        }.items() if v is not None}
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            safety_checker=None,
        )
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.pipe.to(self.device)

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        with self._lock:
            generator = torch.Generator(self.device).manual_seed(self.seed)
            result = self.pipe(
                instruction,
                image=image,
                generator=generator,
                **self.pipeline_kwargs,
            )
            return result.images[0]


class LongCatImageEditModel(BaseModel):
    """meituan-longcat/LongCat-Image-Edit via diffusers LongCatImageEditPipeline.

    Uses enable_model_cpu_offload() by default (~18 GB VRAM).
    Pass high_vram=True to skip offloading for faster inference on large GPUs.
    """

    MODEL_ID = "meituan-longcat/LongCat-Image-Edit"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str      = "bfloat16",
        seed: int             = 42,
        high_vram: bool       = False,
        num_inference_steps: Optional[int]  = None,
        guidance_scale: Optional[float]     = None,
        negative_prompt: Optional[str]      = None,
        **_,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype     = getattr(torch, torch_dtype)
        self.seed      = seed
        self.high_vram = high_vram
        self.pipeline_kwargs = {k: v for k, v in {
            "num_inference_steps": num_inference_steps,
            "guidance_scale":      guidance_scale,
            "negative_prompt":     negative_prompt,
        }.items() if v is not None}
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from diffusers import LongCatImageEditPipeline
        self.pipe = LongCatImageEditPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
        )
        if self.high_vram:
            self.pipe.to(self.device)
        else:
            self.pipe.enable_model_cpu_offload()

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        with self._lock:
            # Generator must be on CPU when using enable_model_cpu_offload
            generator_device = self.device if self.high_vram else "cpu"
            generator = torch.Generator(generator_device).manual_seed(self.seed)
            result = self.pipe(
                image,
                instruction,
                generator=generator,
                **self.pipeline_kwargs,
            )
            return result.images[0]


class QwenImageEditModel(BaseModel):
    """Qwen/Qwen-Image-Edit-2511 via diffusers QwenImageEditPlusPipeline."""

    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str      = "bfloat16",
        seed: int             = 42,
        num_inference_steps: Optional[int]  = None,
        guidance_scale: Optional[float]     = None,
        true_cfg_scale: Optional[float]     = None,
        negative_prompt: str                = " ",  # unconditional conditioning text expected by this model
        **_,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = getattr(torch, torch_dtype)
        self.seed   = seed
        self.pipeline_kwargs = {k: v for k, v in {
            "num_inference_steps": num_inference_steps,
            "guidance_scale":      guidance_scale,
            "true_cfg_scale":      true_cfg_scale,
        }.items() if v is not None}
        self.pipeline_kwargs["negative_prompt"] = negative_prompt
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from diffusers import QwenImageEditPlusPipeline
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        with self._lock:
            generator = torch.Generator(self.device).manual_seed(self.seed)
            with torch.inference_mode():
                output = self.pipe(
                    image=image,
                    prompt=instruction,
                    generator=generator,
                    **self.pipeline_kwargs,
                )
            return output.images[0]


class Flux2DevModel(BaseModel):
    """black-forest-labs/FLUX.2-dev context-based image editing via diffusers Flux2Pipeline."""

    MODEL_ID = "black-forest-labs/FLUX.2-dev"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str      = "bfloat16",
        seed: int             = 42,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float]    = None,
        **_,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = getattr(torch, torch_dtype)
        self.seed   = seed
        self.pipeline_kwargs = {k: v for k, v in {
            "num_inference_steps": num_inference_steps,
            "guidance_scale":      guidance_scale,
        }.items() if v is not None}
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from diffusers import Flux2Pipeline
        self.pipe = Flux2Pipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        with self._lock:
            generator = torch.Generator(self.device).manual_seed(self.seed)
            result = self.pipe(
                prompt=instruction,
                image=[image],
                generator=generator,
                **self.pipeline_kwargs,
            )
            return result.images[0]


class Flux1KontextDevModel(BaseModel):
    """black-forest-labs/FLUX.1-Kontext-dev — BFL's *dedicated* instruction
    editor — via diffusers FluxKontextPipeline.

    Distinct from Flux2DevModel: Kontext is purpose-trained for
    image-conditioned editing (single source image + edit instruction),
    whereas FLUX.2-dev is a flow-matching T2I generator that happens to
    accept image conditioning. This is the model `references.bib::flux2025kontext`
    actually cites — adding this row resolves the citation/model mismatch
    flagged in the 5/5 model coverage audit.

    Gated repo on HF; needs accepted license + HF auth (`hf auth login`).
    """

    MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str      = "bfloat16",
        seed: int             = 42,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float]    = None,
        true_cfg_scale: Optional[float]    = None,
        **_,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = getattr(torch, torch_dtype)
        self.seed   = seed
        self.pipeline_kwargs = {k: v for k, v in {
            "num_inference_steps": num_inference_steps,
            "guidance_scale":      guidance_scale,
            "true_cfg_scale":      true_cfg_scale,
        }.items() if v is not None}
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from diffusers import FluxKontextPipeline
        self.pipe = FluxKontextPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        with self._lock:
            generator = torch.Generator(self.device).manual_seed(self.seed)
            result = self.pipe(
                image=image,
                prompt=instruction,
                generator=generator,
                **self.pipeline_kwargs,
            )
            return result.images[0]


class Flux2Klein9bModel(BaseModel):
    """black-forest-labs/FLUX.2-klein-9B via diffusers Flux2KleinPipeline.

    Distilled 9B sibling of FLUX.2-dev (12B), trained from the same Flux.2
    family but using a separate Pipeline class (Flux2KleinPipeline rather
    than Flux2Pipeline). Higher arena Elo than klein-4B (1223 vs 1189), and
    on par with FLUX.2-dev (1223) per arena.ai single-image-edit ranking
    (May 2026).

    Gated repo on HF; needs accepted license + HF auth (`hf auth login`).
    """

    MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str      = "bfloat16",
        seed: int             = 42,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float]    = None,
        **_,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = getattr(torch, torch_dtype)
        self.seed   = seed
        self.pipeline_kwargs = {k: v for k, v in {
            "num_inference_steps": num_inference_steps,
            "guidance_scale":      guidance_scale,
        }.items() if v is not None}
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from diffusers import Flux2KleinPipeline
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
        )
        self.pipe.to(self.device)

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        with self._lock:
            generator = torch.Generator(self.device).manual_seed(self.seed)
            result = self.pipe(
                prompt=instruction,
                image=[image],
                generator=generator,
                **self.pipeline_kwargs,
            )
            return result.images[0]


# ─── BAGEL upstream pin ───────────────────────────────────────────────────────
# BAGEL doesn't ship as a pip package — upstream
# https://github.com/bytedance-seed/BAGEL has no setup.py / pyproject.toml,
# and the HF model card (ByteDance-Seed/BAGEL-7B-MoT) is weights-only with
# no modeling .py files, so trust_remote_code=True won't work either.
# Rather than vendoring ~5.6 kLOC across modeling/ + data/ + inferencer.py,
# BAGELModel.load_model() auto-clones the upstream tree to a HF-style cache
# dir and adds it to sys.path. The clone is pinned to the SHA below so
# inference is reproducible; bump deliberately + GPU-smoke before merging.
_BAGEL_UPSTREAM_REPO = "https://github.com/bytedance-seed/BAGEL.git"
_BAGEL_UPSTREAM_REF  = "a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f"  # main, 2026-05-04


class BAGELModel(BaseModel):
    """ByteDance-Seed/BAGEL-7B-MoT via the upstream BAGEL inference code.

    BAGEL is a research-code drop, not a pip package — see the comment
    above ``_BAGEL_UPSTREAM_REF`` for why ``trust_remote_code`` and
    ``pip install git+...`` both don't work. ``load_model()`` therefore
    auto-clones https://github.com/bytedance-seed/BAGEL.git pinned at
    ``_BAGEL_UPSTREAM_REF`` to a HF-style cache dir
    (``<HF cache>/paintbench/bagel-upstream``, resolved via
    :meth:`_default_upstream_cache_dir`) on first use and adds it to
    ``sys.path``. Subsequent loads with the cache already populated at
    the pinned SHA are a no-op.

    Air-gapped / offline use: point at an existing upstream checkout
    via ``--bagel-repo <path>`` on the CLI or ``BAGEL_REPO`` in the
    env. When either is set the auto-clone is skipped entirely; the
    path is trusted verbatim (no SHA check), so it's the user's job to
    keep that checkout in sync with the pin if reproducibility matters.

    Inference uses upstream's ``InterleaveInferencer`` and needs
    ``flash_attn`` on the GPU node — install separately on the cluster,
    this wrapper doesn't pull it (no PyPI wheel that matches every
    torch/CUDA combo, and ``--no-build-isolation`` is finicky in CI).

    Chain-of-thought (think=True by default) is saved as a .txt file
    next to every output image, as per the existing reasoning-text
    convention.
    """

    MODEL_ID = "ByteDance-Seed/BAGEL-7B-MoT"
    _LOCAL_DIR_NAME = "bagel-upstream"

    @classmethod
    def _default_upstream_cache_dir(cls) -> str:
        """Resolve the default location for the auto-cloned upstream tree.

        Mirrors :meth:`HunyuanImage3InstructDistilModel._default_model_cache_dir`
        — same HF cache chain (``HF_HUB_CACHE`` /
        ``HUGGINGFACE_HUB_CACHE`` / ``${HF_HOME}/hub`` /
        ``${XDG_CACHE_HOME}/huggingface/hub`` /
        ``~/.cache/huggingface/hub``), with the BAGEL clone landing in
        ``<HF cache>/paintbench/bagel-upstream`` so it sits next to
        other paintbench-owned cache subtrees and inherits whatever
        fast/large filesystem the HF stack already uses.

        Read at call time (not module import) so a ``.env`` or
        ``monkeypatch.setenv`` after import is honoured — same
        rationale as Hunyuan's helper.
        """
        hub_cache = (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
        )
        if not hub_cache:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hub_cache = os.path.join(hf_home, "hub")
            else:
                xdg_cache = os.environ.get("XDG_CACHE_HOME")
                cache_root = xdg_cache or os.path.join(
                    os.path.expanduser("~"), ".cache",
                )
                hub_cache = os.path.join(cache_root, "huggingface", "hub")
        return os.path.join(hub_cache, "paintbench", cls._LOCAL_DIR_NAME)

    @classmethod
    def _ensure_upstream(cls, target_dir: str) -> str:
        """Make ``target_dir`` an upstream BAGEL checkout at the pinned SHA.

        Idempotent: if ``target_dir/.git`` exists and HEAD already matches
        ``_BAGEL_UPSTREAM_REF``, this is a single ``git rev-parse`` and
        returns immediately. Otherwise clones (when empty) or fetches +
        detaches HEAD onto the pin (when present at a different SHA).

        Race-safety is best-effort — concurrent calls from two processes
        on the same node could trip an intermediate state. Typical
        workflow is one inference process per SLURM job, so the window
        is small; on observed breakage, delete the cache dir and rerun.
        """
        target = Path(target_dir).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)

        if shutil.which("git") is None:
            raise RuntimeError(
                "git not found on PATH — required for auto-cloning BAGEL "
                "upstream. Either install git, or pre-populate the upstream "
                f"checkout at {target} and point at it with --bagel-repo / "
                "BAGEL_REPO."
            )

        def _git(*args):
            try:
                return subprocess.run(
                    ["git", *args],
                    check=True, capture_output=True, text=True,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"git {' '.join(args)} failed (exit {exc.returncode}): "
                    f"{(exc.stderr or exc.stdout or '').strip()}"
                ) from exc

        if (target / ".git").exists():
            current = _git("-C", str(target), "rev-parse", "HEAD").stdout.strip()
            if current == _BAGEL_UPSTREAM_REF:
                return str(target)
            print(
                f"  BAGEL upstream at {target} is at {current[:8]}; "
                f"updating to {_BAGEL_UPSTREAM_REF[:8]}"
            )
            _git("-C", str(target), "fetch", "--depth=1", "origin", _BAGEL_UPSTREAM_REF)
            _git("-C", str(target), "checkout", "--detach", "--quiet", _BAGEL_UPSTREAM_REF)
            return str(target)

        print(
            f"  Cloning BAGEL upstream {_BAGEL_UPSTREAM_REPO}"
            f"@{_BAGEL_UPSTREAM_REF[:8]} -> {target}"
        )
        # Clone into a sibling staging dir first, then atomically rename
        # onto `target`. If any step below fails (network blip, killed
        # mid-fetch on first cluster smoke, etc.), `target` is left
        # untouched — the half-written tree lives at `target.tmp` and
        # gets blown away by the rmtree on the next attempt. This avoids
        # the unborn-HEAD trap where `git rev-parse HEAD` on a re-entry
        # exits 128 with an opaque "ambiguous argument 'HEAD'" error.
        staging = target.with_name(target.name + ".tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        _git("init", "--quiet", str(staging))
        _git("-C", str(staging), "remote", "add", "origin", _BAGEL_UPSTREAM_REPO)
        _git("-C", str(staging), "fetch", "--depth=1", "--quiet", "origin", _BAGEL_UPSTREAM_REF)
        _git("-C", str(staging), "checkout", "--detach", "--quiet", _BAGEL_UPSTREAM_REF)
        os.replace(staging, target)
        return str(target)

    def __init__(
        self,
        device: Optional[str] = None,
        torch_dtype: str = "bfloat16",
        seed: int = 42,
        bagel_repo: Optional[str] = None,
        max_memory_gb_per_gpu: Optional[int] = None,
        think: bool = True,
        max_think_token_n: int = 1024,
        num_inference_steps: Optional[int] = None,   # → num_timesteps
        cfg_text_scale: Optional[float] = None,
        cfg_img_scale: Optional[float] = None,
        timestep_shift: Optional[float] = None,
        **_,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, torch_dtype)
        self.seed = seed
        # bagel_repo defaults to None — load_model() resolves it lazily,
        # auto-cloning the pinned upstream into a HF-style cache dir when
        # no explicit override (kwarg / --bagel-repo / BAGEL_REPO) is set.
        self.bagel_repo = bagel_repo
        self.max_memory_gb_per_gpu = max_memory_gb_per_gpu
        self.think = think
        self.max_think_token_n = max_think_token_n

        self.inference_hyper: Dict = {
            "cfg_text_scale":  cfg_text_scale  if cfg_text_scale  is not None else 4.0,
            "cfg_img_scale":   cfg_img_scale   if cfg_img_scale   is not None else 2.0,
            "cfg_interval":    [0.0, 1.0],
            "timestep_shift":  timestep_shift  if timestep_shift  is not None else 3.0,
            "num_timesteps":   num_inference_steps if num_inference_steps is not None else 50,
            "cfg_renorm_min":  0.0,
            "cfg_renorm_type": "text_channel",
        }
        if think:
            self.inference_hyper["max_think_token_n"] = max_think_token_n
            self.inference_hyper["do_sample"] = False

        # Expose inference params as pipeline_kwargs so get_model_info() logs them.
        self.pipeline_kwargs = dict(self.inference_hyper, think=think)
        self.pipe = None  # set to InterleaveInferencer after load_model()
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def _resolve_upstream_dir(self) -> str:
        """Resolve where the upstream BAGEL checkout lives.

        Precedence: ``__init__`` kwarg (= ``--bagel-repo``) >
        ``BAGEL_REPO`` env > auto-cloned upstream at the pinned SHA in
        the HF-style cache. Memoises the resolved path on
        ``self.bagel_repo`` so subsequent calls + ``get_model_info()``
        observe the same value, and so test code can introspect what
        ``load_model`` would have used. Split out from ``load_model``
        so it's testable without GPU / weights / accelerate imports.
        """
        upstream_dir = (
            self.bagel_repo
            or os.environ.get("BAGEL_REPO")
            or self._ensure_upstream(self._default_upstream_cache_dir())
        )
        self.bagel_repo = upstream_dir
        return upstream_dir

    def load_model(self) -> None:
        upstream_dir = self._resolve_upstream_dir()
        print(f"  BAGEL upstream: {upstream_dir}")

        if upstream_dir not in sys.path:
            sys.path.insert(0, upstream_dir)

        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
        from huggingface_hub import snapshot_download
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer
        from data.transforms import ImageTransform
        from data.data_utils import add_special_tokens
        from modeling.autoencoder import load_ae
        from inferencer import InterleaveInferencer
        import modeling.qwen2.modeling_qwen2 as _qwen2_mod

        # Newer transformers versions dropped the "default" rope key that BAGEL expects.
        # Patch it in if missing so model init doesn't KeyError.
        if "default" not in _qwen2_mod.ROPE_INIT_FUNCTIONS:
            def _default_rope_init(config, device=None, **kwargs):
                base = getattr(config, "rope_theta", 10000.0)
                dim = config.hidden_size // config.num_attention_heads
                inv_freq = 1.0 / (
                    base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
                )
                return inv_freq, 1.0
            _qwen2_mod.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init

        model_path = snapshot_download(self.MODEL_ID)

        llm_config = Qwen2Config.from_json_file(
            os.path.join(model_path, "llm_config.json")
        )
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"
        llm_config.pad_token_id = getattr(llm_config, "pad_token_id", None)

        vit_config = SiglipVisionConfig.from_json_file(
            os.path.join(model_path, "vit_config.json")
        )
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_path, "ae.safetensors")
        )

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model      = SiglipVisionModel(vit_config)
            model          = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        num_gpus = torch.cuda.device_count() if HAS_TORCH else 0
        mem_per_gpu = f"{self.max_memory_gb_per_gpu}GiB" if self.max_memory_gb_per_gpu else "80GiB"
        max_memory = {i: mem_per_gpu for i in range(max(num_gpus, 1))}

        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory,
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        # Ensure embedding and projection modules share a device with the rest
        # of the model (required for correct hook-based dispatch).
        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        if num_gpus == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device
        else:
            first_device = device_map.get(same_device_modules[0])
            for k in same_device_modules:
                if k in device_map:
                    device_map[k] = first_device

        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(model_path, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=self.dtype,
            force_hooks=True,
        ).eval()

        tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        self.pipe = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=ImageTransform(1024, 512, 16),
            vit_transform=ImageTransform(980, 224, 14),
            new_token_ids=new_token_ids,
        )

    def generate(self, image: Image.Image, instruction: str) -> ModelOutput:
        with self._lock:
            torch.manual_seed(self.seed)
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)

            output_dict = self.pipe(
                image=image,
                text=instruction,
                think=self.think,
                **self.inference_hyper,
            )

            output_image: Image.Image = output_dict["image"]
            reasoning_text: Optional[str] = output_dict.get("text") or None

            if reasoning_text:
                return output_image, reasoning_text
            return output_image


class NanoBanana2Model(BaseModel):
    """Google Nano Banana 2 (Gemini 3.1 Flash Image Preview) via Gemini API.

    Requires API key via --api-key or GEMINI_API_KEY / GOOGLE_API_KEY env var.
    Supports parallel async inference via generate_async() + concurrency param.

    Configuration layering: kwarg > <ENV_KEY>_<NAME> env > GEMINI_<NAME>
    env > class default. The per-variant ``<ENV_KEY>_*`` layer is what lets
    NB1 + NB2 share one .env (one gateway URL, one API key) while still
    routing each registry key to its own underlying Gemini model id —
    e.g. NB2_MODEL_NAME and NB1_MODEL_NAME can both be set without
    interfering. ``GEMINI_*`` env vars remain a global fallback for back-
    compat with single-variant setups.

    Optional environment variables (all lowercase-canonical defaults
    apply when unset):

        GEMINI_BASE_URL                Alternative base URL for the SDK
                                       client. When set, the client uses
                                       the standard Generative Language
                                       API mode rather than vertexai=True,
                                       since gateways typically don't
                                       support Vertex auth flows. Shared
                                       across variants — same gateway.
        <ENV_KEY>_MODEL_NAME           Override the model id for this
        GEMINI_MODEL_NAME              variant; gateways often expose
                                       Gemini under different names.
        <ENV_KEY>_INCLUDE_THOUGHTS     Override include_thoughts for this
        GEMINI_INCLUDE_THOUGHTS        variant. Falsy: "0", "false", "no", "".
    """

    # Subclasses override these three to define a new variant. Defaults
    # are intentionally canonical Google names — gateway-specific aliases
    # belong in .env (gitignored), not in tracked code.
    DEFAULT_MODEL_NAME = "gemini-3.1-flash-image-preview"
    DEFAULT_INCLUDE_THOUGHTS = True
    ENV_KEY = "NB2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        include_thoughts: Optional[bool] = None,
        concurrency: int = 4,
        **_,
    ):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key required. Pass --api-key or set GEMINI_API_KEY/GOOGLE_API_KEY env var"
                )
        if not model_name:
            model_name = (
                os.getenv(f"{self.ENV_KEY}_MODEL_NAME")
                or os.getenv("GEMINI_MODEL_NAME")
                or self.DEFAULT_MODEL_NAME
            )
        if not base_url:
            base_url = os.getenv("GEMINI_BASE_URL")  # None => Vertex AI default
        if include_thoughts is None:
            raw = (
                os.getenv(f"{self.ENV_KEY}_INCLUDE_THOUGHTS")
                or os.getenv("GEMINI_INCLUDE_THOUGHTS")
            )
            if raw is None:
                include_thoughts = self.DEFAULT_INCLUDE_THOUGHTS
            else:
                include_thoughts = raw.lower() not in ("0", "false", "no", "")
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.include_thoughts = include_thoughts
        self.concurrency = concurrency
        self.client = None
        self.MODEL_ID = model_name

    def load_model(self) -> None:
        try:
            from google import genai
        except ImportError:
            raise ImportError("google-genai required. Install: pip install google-genai")
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["http_options"] = {
                "api_version": "v1",
                "base_url": self.base_url,
            }
        else:
            client_kwargs["vertexai"] = True
        self.client = genai.Client(**client_kwargs)
        endpoint = self.base_url or "vertex-ai-model-garden"
        print(f"Gemini API initialized (model: {self.model_name}, endpoint: {endpoint}, concurrency={self.concurrency})")

    def _parse_response(self, response, types) -> Tuple[Image.Image, str]:
        if response.candidates[0].finish_reason != types.FinishReason.STOP:
            raise RuntimeError(f"Generation failed: {response.candidates[0].finish_reason}")

        output_image = None
        reasoning_parts = []

        for part in response.candidates[0].content.parts:
            if part.thought:
                reasoning_parts.append(f"[THINKING] {part.text}")
            elif part.inline_data:
                output_image = Image.open(io.BytesIO(part.inline_data.data))
            elif part.text:
                reasoning_parts.append(f"[REASONING] {part.text}")

        if output_image is None:
            raise ValueError("No image in response.")

        reasoning_text = "\n\n".join(reasoning_parts) or "No reasoning trace captured."
        return output_image, reasoning_text

    def generate(self, image: Image.Image, instruction: str) -> Tuple[Image.Image, str]:
        if self.client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from google.genai import types

        config_kwargs = {"response_modalities": ["IMAGE", "TEXT"]}
        if self.include_thoughts:
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        config = types.GenerateContentConfig(**config_kwargs)

        deadline = None
        last_exc = None
        while True:
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[image, instruction],
                    config=config,
                )
                break
            except Exception as exc:
                if not _is_transient_error(exc):
                    raise
                if deadline is None:
                    deadline = time.perf_counter() + 120
                    print(f"      Transient error, retrying for up to 2 min: {type(exc).__name__}")
                last_exc = exc
                if time.perf_counter() >= deadline:
                    raise last_exc
                time.sleep(2)

        return self._parse_response(response, types)

    async def generate_async(self, image: Image.Image, instruction: str) -> Tuple[Image.Image, str]:
        if self.client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from google.genai import types

        # Mirrors the sync generate(): include_thoughts is conditional (NB1
        # rejects thinking_config), and the retry predicate covers transports
        # not just 429s -- async runs hit the same gateway flakiness as sync.
        config_kwargs = {"response_modalities": ["IMAGE", "TEXT"]}
        if self.include_thoughts:
            config_kwargs["thinking_config"] = types.ThinkingConfig(include_thoughts=True)
        config = types.GenerateContentConfig(**config_kwargs)

        deadline = None
        last_exc = None
        while True:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_name,
                    contents=[image, instruction],
                    config=config,
                )
                break
            except Exception as exc:
                if not _is_transient_error(exc):
                    raise
                if deadline is None:
                    deadline = time.perf_counter() + 120
                    print(f"      Transient error, retrying for up to 2 min: {type(exc).__name__}")
                last_exc = exc
                if time.perf_counter() >= deadline:
                    raise last_exc
                await asyncio.sleep(2)

        return self._parse_response(response, types)


class GptImage2Model(BaseModel):
    """OpenAI GPT-Image-2 via the OpenAI Images edits API.

    Requires API key via --api-key or OPENAI_API_KEY env var.
    Supports parallel async inference via generate_async() + concurrency param.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-image-2",
        size: str = "auto",
        concurrency: int = 8,
        **_,
    ):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "API key required. Pass --api-key or set OPENAI_API_KEY env var"
                )
        self.api_key = api_key
        self.model_name = model_name
        self.size = size
        self.concurrency = concurrency
        self.client = None
        self.async_client = None

    def load_model(self) -> None:
        try:
            from openai import AsyncOpenAI, OpenAI
        except ImportError:
            raise ImportError("openai required. Run: pip install openai")
        self.client = OpenAI(api_key=self.api_key)
        self.async_client = AsyncOpenAI(api_key=self.api_key)
        print(f"OpenAI Images API initialized (model: {self.model_name}, concurrency={self.concurrency})")

    # gpt-image-2 requires at least 655,360 total pixels.
    _MIN_PIXELS = 655_360

    def _resolve_size(self, image: Image.Image) -> str:
        if self.size != "auto":
            return self.size
        w = ((image.width  + 15) // 16) * 16
        h = ((image.height + 15) // 16) * 16
        # Scale up proportionally if below the API's minimum pixel budget.
        if w * h < self._MIN_PIXELS:
            import math
            scale = math.sqrt(self._MIN_PIXELS / (w * h))
            w = ((math.ceil(w * scale) + 15) // 16) * 16
            h = ((math.ceil(h * scale) + 15) // 16) * 16
        return f"{w}x{h}"

    def generate(self, image: Image.Image, instruction: str) -> Image.Image:
        if self.client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import base64

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        size = self._resolve_size(image)

        deadline = None
        last_exc = None
        while True:
            try:
                response = self.client.images.edit(
                    model=self.model_name,
                    image=("input.png", io.BytesIO(image_bytes), "image/png"),
                    prompt=instruction,
                    n=1,
                    size=size,
                )
                break
            except Exception as exc:
                if "429" not in str(exc):
                    raise
                if deadline is None:
                    deadline = time.perf_counter() + 120
                    print("      429 rate limit hit, retrying for up to 2 min...")
                last_exc = exc
                if time.perf_counter() >= deadline:
                    raise last_exc
                time.sleep(5)

        output_bytes = base64.b64decode(response.data[0].b64_json)
        return Image.open(io.BytesIO(output_bytes))

    async def generate_async(self, image: Image.Image, instruction: str) -> Image.Image:
        if self.async_client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import base64

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        size = self._resolve_size(image)

        deadline = None
        last_exc = None
        while True:
            try:
                response = await self.async_client.images.edit(
                    model=self.model_name,
                    image=("input.png", io.BytesIO(image_bytes), "image/png"),
                    prompt=instruction,
                    n=1,
                    size=size,
                )
                break
            except Exception as exc:
                if "429" not in str(exc):
                    raise
                if deadline is None:
                    deadline = time.perf_counter() + 120
                    print("      429 rate limit hit, retrying for up to 2 min...")
                last_exc = exc
                if time.perf_counter() >= deadline:
                    raise last_exc
                await asyncio.sleep(5)

        output_bytes = base64.b64decode(response.data[0].b64_json)
        return Image.open(io.BytesIO(output_bytes))


class NanoBanana1Model(NanoBanana2Model):
    """Google Nano Banana 1 (Gemini 2.5 Flash Image) via Gemini API.

    Earlier variant — Vertex rejects ``Thinking_config.include_thoughts``
    for this model, so the class default is False. Same env layering as
    NanoBanana2Model; reads ``NB1_MODEL_NAME`` / ``NB1_INCLUDE_THOUGHTS``
    for per-variant overrides (typically the gateway-specific model id).
    """

    DEFAULT_MODEL_NAME = "gemini-2.5-flash-image"
    DEFAULT_INCLUDE_THOUGHTS = False
    ENV_KEY = "NB1"


class HunyuanImage3InstructDistilModel(BaseModel):
    """tencent/HunyuanImage-3.0-Instruct-Distil — Tencent's 80B MoE (13B active)
    native multimodal model, distilled for 8-step sampling.

    #1 open-weights editor on both arena.ai (Elo 1301, rank 16 overall)
    and Artificial Analysis (Elo 1226) image-edit leaderboards as of May 2026.

    Uses transformers AutoModelForCausalLM with trust_remote_code=True.
    The model's custom ``generate_image()`` method handles the full pipeline:
    tokenization → optional CoT reasoning → diffusion generation.

    **VRAM: ≥ 8 × 80 GB** (model card recommendation). The 80B MoE params
    are sharded across GPUs via ``device_map="auto"``. Single-GPU is not
    feasible. Run on multi-GPU nodes (8×H100/H200/A100-80G).

    **Concurrency: use ``--workers 1``** (default). ``generate()`` holds
    a ``threading.Lock`` (see ``BaseModel`` thread-safety contract), so
    passing ``--workers N > 1`` won't crash and won't oversubscribe VRAM
    — the lock serialises the GPU step, only one thread allocates
    activations at a time. But it also won't *speed up* anything on this
    model: the lock makes the GPU step (already the bottleneck) strictly
    serial, so additional workers only buy CPU-side overlap on
    pre/post-processing — negligible vs. an 80B MoE forward pass. Skip
    the extra thread overhead and the 640 GB activation footprint's
    fragility margin; keep ``--workers 1``. Same reasoning applies to
    BAGEL, but is especially load-bearing here.

    The model path must not contain dots (HF ``trust_remote_code`` import
    limitation), so ``load_model()`` downloads to a sanitized local dir
    rather than using HF's normal ``models--<org>--<name>/snapshots/<rev>``
    layout (the revision ``.``-separated SHA segments break the dynamic
    module import). Default download root is ``<HF cache>/paintbench/<dir>``,
    where ``<HF cache>`` resolves the same way ``huggingface_hub`` does —
    see ``_default_model_cache_dir`` for the full resolution chain
    (``HF_HUB_CACHE`` / ``HUGGINGFACE_HUB_CACHE`` / ``HF_HOME`` /
    ``XDG_CACHE_HOME`` / ``~/.cache``) — so it inherits whatever
    large-storage location the rest of the HF stack already uses. The ``paintbench/`` prefix keeps these sanitized
    snapshots clearly separate from HF's own ``models--…`` directories
    and lets both Hunyuan variants coexist underneath.

    Override precedence is ``model_cache_dir`` kwarg > ``HUNYUAN_MODEL_DIR``
    env > default (e.g. point at a fast local NVMe on a SLURM node).
    ``--model-cache-dir`` on the CLI forwards into the kwarg.
    """

    MODEL_ID = "tencent/HunyuanImage-3.0-Instruct-Distil"
    _LOCAL_DIR_NAME = "HunyuanImage-3-Instruct-Distil"

    @classmethod
    def _default_model_cache_dir(cls) -> str:
        """Resolve the default download root for this variant.

        Returns ``<HF cache>/paintbench/<_LOCAL_DIR_NAME>``, where
        ``<HF cache>`` mirrors ``huggingface_hub``'s resolution rules
        but reads the environment **at call time** (not at module
        import — ``huggingface_hub.constants.HF_HUB_CACHE`` is
        snapshotted on first import and won't reflect env vars set by
        ``.env`` files or test ``monkeypatch.setenv`` after that, which
        is exactly the cluster/test config-injection pattern we want to
        honour). The priority chain is:

        1. ``HF_HUB_CACHE`` env var
        2. ``HUGGINGFACE_HUB_CACHE`` env var (legacy upstream alias,
           still respected for backwards compatibility — long-standing
           user configs and shared cluster module files often only set
           the old name)
        3. ``${HF_HOME}/hub`` if ``HF_HOME`` is set
        4. ``${XDG_CACHE_HOME}/huggingface/hub`` if ``XDG_CACHE_HOME``
           is set (matches huggingface_hub's own resolution — clusters
           and containers that redirect application caches via the XDG
           base-dir spec rather than HF-specific vars still need to
           land in the right cache, otherwise the ~160 GB Hunyuan
           snapshot lands in $HOME and defeats the redirect)
        5. ``~/.cache/huggingface/hub`` (final fallback, same as HF's
           own default)

        The lookup stays inside the method (not at module import) so
        ``import inference`` remains cheap for code paths that never
        touch Hunyuan.
        """
        hub_cache = (
            os.environ.get("HF_HUB_CACHE")
            or os.environ.get("HUGGINGFACE_HUB_CACHE")
        )
        if not hub_cache:
            hf_home = os.environ.get("HF_HOME")
            if hf_home:
                hub_cache = os.path.join(hf_home, "hub")
            else:
                xdg_cache = os.environ.get("XDG_CACHE_HOME")
                cache_root = xdg_cache or os.path.join(
                    os.path.expanduser("~"), ".cache"
                )
                hub_cache = os.path.join(cache_root, "huggingface", "hub")
        return os.path.join(hub_cache, "paintbench", cls._LOCAL_DIR_NAME)

    def __init__(
        self,
        seed: int = 42,
        diff_infer_steps: int = 8,
        bot_task: str = "image",  # skip CoT prompt-rewrite; "think_recaption" for reasoning
        use_system_prompt: Optional[str] = "en_unified",
        moe_impl: str = "eager",
        model_cache_dir: Optional[str] = None,
        **_,
    ):
        self.seed = seed
        self.diff_infer_steps = diff_infer_steps
        self.bot_task = bot_task
        self.use_system_prompt = use_system_prompt
        self.moe_impl = moe_impl
        self.model_cache_dir = model_cache_dir
        self.pipeline_kwargs = {
            "diff_infer_steps": diff_infer_steps,
            "bot_task": bot_task,
            "use_system_prompt": use_system_prompt,
            "moe_impl": moe_impl,
        }
        self.pipe = None
        self._lock = threading.Lock()  # serializes generate(); see BaseModel

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM
        from huggingface_hub import snapshot_download

        cache_root = (
            self.model_cache_dir
            or os.environ.get("HUNYUAN_MODEL_DIR")
            or self._default_model_cache_dir()
        )
        # Surface the resolved location so a misconfigured cache (e.g.
        # an unset HF_HOME silently filling up $HOME) is visible at
        # load time rather than after the download has already burned
        # 160 GB on the wrong filesystem.
        print(f"  Hunyuan cache root: {cache_root}")

        # Skip snapshot_download when the local cache is already complete
        # AND matches the variant we're about to load.
        #
        # Two failure modes this guards against:
        #
        # 1) Concurrent SLURM jobs colliding on per-file locks under
        #    ``cache_root/.cache/huggingface/`` and failing with
        #    ``OSError: [Errno 116] Stale file handle`` on shared
        #    filesystems. Skipping when the cache is already populated
        #    avoids the lock entirely.
        #
        # 2) **Wrong-variant cache poisoning**. If the user follows the docstring
        #    and points ``HUNYUAN_MODEL_DIR`` at a shared path, the first
        #    Hunyuan run populates that path with one variant's weights
        #    and a subsequent run of the *other* variant (e.g. Distil →
        #    Full Instruct) would see shards-present, skip download, and
        #    silently load the wrong weights while the metrics JSON
        #    reports the requested ``MODEL_ID``. To prevent this, we
        #    write a sentinel containing ``MODEL_ID`` after every
        #    successful download and require the sentinel to match
        #    before trusting the cache. Mismatch ⇒ fall through to
        #    snapshot_download, which is idempotent and refreshes any
        #    files whose ETag has changed (i.e. the other variant's
        #    shards get rewritten to this variant's).
        #
        # Completeness shard check uses ``model.safetensors.index.json``;
        # we only verify presence + non-zero size, not full byte counts
        # (good enough for the dominant failure mode; a killed
        # ``snapshot_download`` mid-write leaves a partial shard that
        # ``from_pretrained`` will then catch).
        import json as _json
        idx_path = os.path.join(cache_root, "model.safetensors.index.json")
        sentinel_path = os.path.join(cache_root, ".paintbench_hunyuan_model_id")
        cache_complete = False

        def _read_sentinel():
            # Mirrors the OSError-tolerant pattern used for ``idx_path``
            # below: an existing-but-unreadable sentinel (rare: shared FS
            # permission quirk) should fall through to a clean download
            # rather than crash the load.
            try:
                with open(sentinel_path) as _fp:
                    return _fp.read().strip()
            except OSError:
                return None

        sentinel_matches = _read_sentinel() == self.MODEL_ID
        if os.path.exists(idx_path) and sentinel_matches:
            try:
                with open(idx_path) as _fp:
                    _idx = _json.load(_fp)
                _shards = set(_idx.get("weight_map", {}).values())
                if _shards and all(
                    os.path.exists(os.path.join(cache_root, _f))
                    and os.path.getsize(os.path.join(cache_root, _f)) > 0
                    for _f in _shards
                ):
                    cache_complete = True
            except (OSError, ValueError):
                cache_complete = False

        if not cache_complete:
            snapshot_download(self.MODEL_ID, local_dir=cache_root)
            # Stamp the sentinel only after a successful download so a
            # download killed mid-write doesn't leave a sentinel that
            # would falsely advertise a complete cache on the next run.
            try:
                with open(sentinel_path, "w") as _fp:
                    _fp.write(self.MODEL_ID)
            except OSError:
                # Best-effort: if the FS rejects the write (read-only
                # mount, quota, etc.) we still let load proceed —
                # snapshot_download succeeded, so the weights are
                # correct; we just lose the cache short-circuit on the
                # next run.
                pass

        self.pipe = AutoModelForCausalLM.from_pretrained(
            cache_root,
            attn_implementation="sdpa",
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            moe_impl=self.moe_impl,
            moe_drop_tokens=True,
        )

        # The published remote code references config.model_version in
        # load_tokenizer() but neither the Distil nor full Instruct
        # config.json defines it. Patch it in to avoid AttributeError.
        if not hasattr(self.pipe.config, "model_version"):
            self.pipe.config.model_version = None

        self.pipe.load_tokenizer(cache_root)

        # The Distil snapshot's modeling code calls
        # ``image_processor.build_img_ratio_slice_logits_proc()`` but the
        # image processor defines the method as
        # ``build_img_ratio_slice_logits_processor()`` (full suffix).
        # Alias the truncated name so generate_image() doesn't crash.
        ip = getattr(self.pipe, "image_processor", None)
        if ip is not None and not hasattr(ip, "build_img_ratio_slice_logits_proc"):
            ip.build_img_ratio_slice_logits_proc = getattr(
                ip, "build_img_ratio_slice_logits_processor", None
            )

        # The ViT processor (Siglip2ImageProcessor) may return plain lists
        # for ``pixel_values`` instead of tensors in newer transformers
        # versions. The downstream ``vit_process_image`` then crashes on
        # ``.squeeze(0)`` because lists don't have that method.
        # Monkey-patch ``vit_process_image`` to request PyTorch tensors
        # from the processor directly.
        if ip is not None:
            _orig_vit_process = ip.vit_process_image

            def _patched_vit_process(image):
                origin_size = image.size
                inputs = ip.vit_info.processor(image, return_tensors="pt")
                image_t = inputs["pixel_values"].squeeze(0)

                remain_kwargs = {}
                for key in set(inputs.keys()) - {"pixel_values"}:
                    v = inputs[key]
                    if hasattr(v, "squeeze"):
                        remain_kwargs[key] = v.squeeze(0)
                    else:
                        remain_kwargs[key] = v

                return ip.as_image_tensor(
                    image_t,
                    image_type=ip.vit_info.image_type,
                    origin_size=origin_size,
                    **remain_kwargs,
                )

            ip.vit_process_image = _patched_vit_process

        # The model's custom ``HunyuanStaticCache.update()`` calls
        # ``lazy_initialization(key_states)`` with only one arg, but the
        # current transformers ``StaticLayer.lazy_initialization()``
        # requires both ``key_states`` and ``value_states``.
        # Monkey-patch the class method to forward both arguments.
        import sys
        mod = sys.modules.get(type(self.pipe).__module__)
        _HSC = getattr(mod, "HunyuanStaticCache", None) if mod else None
        if _HSC is not None:

            def _patched_update(self_cache, key_states, value_states, layer_idx, cache_kwargs=None):
                cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
                if self_cache.layers[layer_idx].keys is None:
                    self_cache.layers[layer_idx].lazy_initialization(key_states, value_states)
                k_out = self_cache.layers[layer_idx].keys
                v_out = self_cache.layers[layer_idx].values

                if cache_position is None:
                    k_out.copy_(key_states)
                    v_out.copy_(value_states)
                else:
                    if cache_position.dim() == 1:
                        k_out.index_copy_(2, cache_position, key_states)
                        v_out.index_copy_(2, cache_position, value_states)
                        if self_cache.dynamic:
                            end = cache_position[-1].item() + 1
                            k_out = k_out[:, :, :end]
                            v_out = v_out[:, :, :end]
                    else:
                        batch_size = cache_position.shape[0]
                        for i in range(batch_size):
                            k_out[i].index_copy_(1, cache_position[i], key_states[i])
                            v_out[i].index_copy_(1, cache_position[i], value_states[i])
                        if self_cache.dynamic:
                            end = cache_position[0, -1].item() + 1
                            k_out = k_out[:, :, :end]
                            v_out = v_out[:, :, :end]
                return k_out, v_out

            _HSC.update = _patched_update

        # Hunyuan's ``_update_model_kwargs_for_generation`` (used in the
        # ``gen_text`` path that backs ``bot_task in {"think", "recaption",
        # "think_recaption", "img_ratio"}``) returns a fresh dict that
        # drops the generic ``use_cache`` key. transformers >=5.0 sets
        # ``model_kwargs["use_cache"] = generation_config.use_cache``
        # before entering ``_sample`` and then re-reads it inside the
        # decode loop (utils.py: ``next_sequence_length = 1 if
        # model_kwargs["use_cache"] else None``), so the second decode
        # iteration crashes with ``KeyError: 'use_cache'``.
        # Wrap the override to forward ``use_cache`` from the input to
        # the output. Idempotent — guarded with an attribute marker so
        # repeated ``load_model()`` calls don't double-wrap.
        _cls = type(self.pipe)
        if not getattr(_cls, "_paintbench_use_cache_patched", False):
            _orig_update_kwargs = _cls._update_model_kwargs_for_generation

            def _patched_update_kwargs(
                self_model,
                outputs,
                model_kwargs,
                is_encoder_decoder=False,
                num_new_tokens=1,
            ):
                new_kwargs = _orig_update_kwargs(
                    self_model,
                    outputs,
                    model_kwargs,
                    is_encoder_decoder=is_encoder_decoder,
                    num_new_tokens=num_new_tokens,
                )
                if "use_cache" in model_kwargs and "use_cache" not in new_kwargs:
                    new_kwargs["use_cache"] = model_kwargs["use_cache"]
                return new_kwargs

            _cls._update_model_kwargs_for_generation = _patched_update_kwargs
            _cls._paintbench_use_cache_patched = True

    def generate(self, image: Image.Image, instruction: str) -> ModelOutput:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f, format="PNG")
            tmp_path = f.name

        try:
            with self._lock:
                cot_text, samples = self.pipe.generate_image(
                    prompt=instruction,
                    image=[tmp_path],
                    seed=self.seed,
                    image_size=f"{image.size[1]}x{image.size[0]}",
                    use_system_prompt=self.use_system_prompt,
                    bot_task=self.bot_task,
                    infer_align_image_size=True,
                    diff_infer_steps=self.diff_infer_steps,
                    verbose=0,
                )
        finally:
            os.unlink(tmp_path)

        output_image = samples[0]
        reasoning_text = cot_text[0] if cot_text and cot_text[0] else ""
        if reasoning_text:
            return output_image, reasoning_text
        return output_image

    def get_model_info(self) -> Dict:
        return {
            "model_id": self.MODEL_ID,
            "pipeline_class": type(self.pipe).__name__ if self.pipe is not None else "HunyuanImage3ForCausalMM",
            "device": "auto (multi-GPU)",
            "torch_dtype": "auto (bf16)",
            "seed": self.seed,
            "pipeline_params": self.pipeline_kwargs,
        }


class HunyuanImage3InstructModel(HunyuanImage3InstructDistilModel):
    """tencent/HunyuanImage-3.0-Instruct — the non-distilled flagship.

    Same architecture/API as the Distil variant; differs only in:
      * No 8-step distillation, so default ``diff_infer_steps=50`` per the
        upstream README.
      * Different HF repo / local snapshot directory name.

    Used to test whether sampling-step distillation (Distil → full) closes
    the ~10x PaintBench gap vs. Flux models.
    """

    MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"
    _LOCAL_DIR_NAME = "HunyuanImage-3-Instruct"

    def __init__(self, **kwargs):
        kwargs.setdefault("diff_infer_steps", 50)
        super().__init__(**kwargs)


_REGISTRY: dict[str, type[BaseModel]] = {
    "instruct-pix2pix":   InstructPix2PixModel,
    "longcat-image-edit": LongCatImageEditModel,
    "qwen-image-edit":    QwenImageEditModel,
    "flux2-dev":          Flux2DevModel,
    "flux1-kontext-dev":  Flux1KontextDevModel,
    "flux2-klein-9b":     Flux2Klein9bModel,
    "bagel":              BAGELModel,
    "nano-banana-1":      NanoBanana1Model,
    "nano-banana-2":      NanoBanana2Model,
    "gpt-image-2":        GptImage2Model,
    "hunyuan-image-3-instruct": HunyuanImage3InstructModel,
    "hunyuan-image-3":    HunyuanImage3InstructDistilModel,
}


def create_evaluator(registry_key: str, api_key: Optional[str] = None, **kwargs) -> BaseModel:
    """Instantiate a model by registry name. api_key is accepted for API-based models.

    The first parameter is named ``registry_key`` (not ``model_name``) so callers
    can forward a constructor-level ``model_name`` kwarg (used by Gemini models
    to override the underlying model id) without colliding with the positional.
    """
    key = registry_key.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown model {registry_key!r}. Available: {available}")
    if api_key is not None:
        kwargs["api_key"] = api_key
    return _REGISTRY[key](**kwargs)


# ─── System / Hardware Info ───────────────────────────────────────────────────

def get_gpu_info() -> List[Dict]:
    """Return hardware properties for each available CUDA GPU."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / 1024 ** 3, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    return gpus


def get_system_info() -> Dict:
    """Collect system-level hardware and software information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "logical_cpu_cores": os.cpu_count(),
        "physical_cpu_cores": None,
        "ram_total_gb": None,
        "torch_version": torch.__version__ if HAS_TORCH else None,
        "cuda_version": torch.version.cuda if HAS_TORCH else None,
        "num_gpus": torch.cuda.device_count() if HAS_TORCH else 0,
        "gpus": get_gpu_info(),
    }

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1024 ** 3, 2)
        info["physical_cpu_cores"] = psutil.cpu_count(logical=False)

    return info


def get_gpu_memory_snapshot() -> Dict:
    """Snapshot current and peak GPU memory usage across all devices."""
    if not HAS_TORCH or not torch.cuda.is_available():
        return {}

    snapshot = {}
    for i in range(torch.cuda.device_count()):
        snapshot[f"gpu_{i}"] = {
            "allocated_gb": round(torch.cuda.memory_allocated(i) / 1024 ** 3, 3),
            "reserved_gb": round(torch.cuda.memory_reserved(i) / 1024 ** 3, 3),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(i) / 1024 ** 3, 3),
        }
    return snapshot


def reset_peak_memory_stats():
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


# ─── Benchmark Loading ────────────────────────────────────────────────────────

def load_benchmark(
    source: BenchmarkSource,
    task_filter: Optional[List[str]] = None,
) -> Dict[str, List[Problem]]:
    """Load problems from a :class:`BenchmarkSource` keyed by task name.

    The actual discovery logic lives in the source implementation
    (:class:`~benchmark_source.LocalBenchmarkSource` for the on-disk
    layout, :class:`~benchmark_source.HfBenchmarkSource` for HF datasets).
    This wrapper just drains the iterator, applies ``task_filter``, and
    drops empty tasks to preserve the pre-refactor return shape.

    Each returned :class:`Problem` supports dict-style access via
    ``__getitem__`` so the downstream orchestrator can keep using
    ``problem["input_image"]`` / ``problem["instruction"]`` / etc.
    """
    tasks: Dict[str, List[Problem]] = {}
    for task in source.iter_tasks():
        if task_filter and task not in task_filter:
            continue
        problems = list(source.iter_problems(task))
        if problems:
            tasks[task] = problems

    if not tasks:
        filter_clause = f", filter={task_filter!r}" if task_filter else ""
        raise FileNotFoundError(
            f"No tasks found for benchmark source {source.name()!r} "
            f"({source.revision()}{filter_clause})"
        )
    return tasks


# ─── Per-Problem Inference ────────────────────────────────────────────────────

def _unpack_generate_result(
    result: Union[Image.Image, Tuple],
) -> Tuple[Optional[Image.Image], Optional[str], Dict[str, bytes]]:
    """Normalise the polymorphic return type of ``BaseModel.generate``.

    Models may return one of:

      - ``image`` alone                        (image-edit models, no reasoning)
      - ``(image, reasoning_text)``            (legacy 2-tuple)
      - ``(image, reasoning_text, sidecars)``  (extended 3-tuple, where
          ``sidecars`` maps a filename-tail string to raw bytes)

    Returns ``(image_or_None, reasoning_or_None, sidecars_dict)`` so the
    orchestrator can write each output uniformly. ``sidecars`` is always a
    dict (possibly empty) so callers can iterate without a None-check.
    """
    if isinstance(result, tuple):
        if len(result) == 3:
            img, reasoning, sidecars = result
            return img, reasoning, dict(sidecars or {})
        if len(result) == 2:
            img, reasoning = result
            return img, reasoning, {}
        raise ValueError(
            f"generate() returned tuple of unexpected length {len(result)}; "
            f"expected a bare Image, (image, reasoning), or "
            f"(image, reasoning, sidecars)"
        )
    return result, None, {}


def _extra_sidecar_path(output_path: Path, suffix: str) -> Path:
    """Compute the on-disk path for a model-supplied extra sidecar.

    Convention: extra sidecars sit next to the output PNG with the
    ``_output`` segment stripped from the stem, then the ``suffix`` tail
    appended. So for ``.../0007_output.png`` and suffix ``"_trace.json"``
    the result is ``.../0007_trace.json``. Falls back to ``with_suffix``
    for tails that start with ``.`` and don't begin with ``_``
    (e.g. ``.bin`` becomes a normal suffix swap).
    """
    if suffix.startswith("_"):
        stem = output_path.stem
        if stem.endswith("_output"):
            stem = stem[: -len("_output")]
        return output_path.with_name(f"{stem}{suffix}")
    return output_path.with_suffix(suffix)


def _write_extra_sidecars(
    output_path: Path, sidecars: Dict[str, bytes]
) -> List[Path]:
    """Write each extra sidecar next to ``output_path``; return paths written.

    Empty inputs short-circuit. Each sidecar's parent dir is created if
    missing. Failures (e.g. disk full) propagate — extras are part of the
    artifact contract for the model that emits them.
    """
    written: List[Path] = []
    if not sidecars:
        return written
    for suffix, content in sidecars.items():
        sidecar_path = _extra_sidecar_path(output_path, suffix)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_bytes(content)
        written.append(sidecar_path)
    return written


def run_single(model, problem: Dict, output_path: Path) -> Dict:
    """
    Run model inference for one problem and save the output image.

    generate() may return either a PIL Image, a (PIL Image, reasoning_text)
    tuple, or a (PIL Image, reasoning_text, extra_sidecars) tuple. If reasoning
    text is present it is saved alongside the output image as a .txt file
    with the same stem; each extra sidecar is written alongside via
    :func:`_extra_sidecar_path` (e.g. ``_trace.json`` → ``<NNNN>_trace.json``).

    Returns a dict of per-sample metrics. Does NOT compute any accuracy scores.
    """
    input_image = problem["input_image"]
    instruction = problem["instruction"]

    reset_peak_memory_stats()
    mem_before = get_gpu_memory_snapshot()

    t_start = time.perf_counter()
    result = model.generate(input_image, instruction)
    t_end = time.perf_counter()

    inference_time = t_end - t_start
    mem_after = get_gpu_memory_snapshot()

    output_image, reasoning_text, extra_sidecars = _unpack_generate_result(result)

    success = output_image is not None
    output_size = None
    reasoning_path = None
    extra_paths: List[Path] = []

    if success:
        output_image = output_image.convert("RGB")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_image.save(output_path)
        output_size = list(output_image.size)  # [width, height]

        if reasoning_text:
            reasoning_path = output_path.with_suffix(".txt")
            reasoning_path.write_text(reasoning_text, encoding="utf-8")

        extra_paths = _write_extra_sidecars(output_path, extra_sidecars)

    return {
        "index": problem["index"],
        "task": problem["task"],
        "mode": problem["mode"],
        "visual_condition": problem["visual_condition"],
        "instruction": instruction,
        "output_path": str(output_path) if success else None,
        "reasoning_path": str(reasoning_path) if reasoning_path else None,
        "reasoning_text": reasoning_text,
        "extra_artifact_paths": [str(p) for p in extra_paths],
        "success": success,
        "inference_time_s": round(inference_time, 4),
        "input_size_wh": list(input_image.size),
        "output_size_wh": output_size,
        "gpu_memory_before": mem_before,
        "gpu_memory_after": mem_after,
    }


# ─── Aggregation Helpers ──────────────────────────────────────────────────────

def _failed_problem_keys(prior_metrics: Dict) -> set:
    """Extract ``(task_id, problem_index)`` tuples for problems that failed
    in a prior run's metrics JSON. Used by ``--retry-failed`` to filter the
    loaded benchmark down to just the problems worth retrying."""
    return {
        (task_id, p["index"])
        for task_id, task in prior_metrics.get("tasks", {}).items()
        for p in task.get("problems", [])
        if not p.get("success")
    }


def _print_corrupt_cache_note(forecast_cached: int, actual_skipped: int) -> None:
    """Print a one-line note when the stat()-based forecast in the Plan
    over-counted cached PNGs — i.e. some files that ``stat()`` saw as
    "cached" turned out to be corrupt or truncated and fell through
    :func:`_build_skipped_result`'s decode check, getting redone. Silent
    when they agree."""
    if forecast_cached and actual_skipped < forecast_cached:
        diff = forecast_cached - actual_skipped
        plural = "" if diff == 1 else "s"
        were = "was" if diff == 1 else "were"
        print(f"  ({diff} forecast-cached PNG{plural} {were} corrupt and rerun)")


def _build_skipped_result(problem: Dict, output_path: Path) -> Optional[Dict]:
    """Build a synthetic result dict for a problem whose output PNG is
    already on disk (and readable). Returns ``None`` when the cache is
    missing, empty, or fails to decode — the caller should fall through
    and re-run the model in those cases.

    Used by the default-incremental rerun path to short-circuit
    problems that have a usable cached output. Validates the cached
    PNG with ``PIL.Image.load()`` so truncated writes from an
    interrupted prior run are detected and redone rather than silently
    accepted. ``--overwrite`` skips this check entirely.
    """
    # Single try/except covers (a) FileNotFoundError from stat() when the
    # cache is missing, (b) zero-byte files from a write that died before
    # any data flushed, and (c) PIL.Image.load() failures from a partial /
    # corrupt PNG. Any of these → fall through and re-run.
    try:
        if output_path.stat().st_size == 0:
            return None
        with Image.open(output_path) as cached:
            cached.load()
            output_size = list(cached.size)
    except Exception:
        return None

    try:
        input_size = list(problem["input_size_wh"])
    except Exception:
        return None

    reasoning_path = output_path.with_suffix(".txt")
    reasoning_text: Optional[str] = None
    if reasoning_path.exists():
        try:
            reasoning_text = reasoning_path.read_text(encoding="utf-8")
        except Exception:
            reasoning_text = None

    # Surface any extra sidecars (model-emitted artifacts named like
    # ``<NNNN>_<suffix>``) that were written next to the cached PNG by an
    # earlier run, so the metrics JSON for a cache-hit looks identical to a
    # fresh-run result.
    extra_paths = sorted(
        output_path.parent.glob(
            output_path.stem.removesuffix("_output") + "_*"
        )
    )
    extra_paths = [
        p for p in extra_paths
        if p.is_file() and p != output_path and p != reasoning_path
    ]

    return {
        "index": problem["index"],
        "task": problem["task"],
        "mode": problem["mode"],
        "visual_condition": problem["visual_condition"],
        "instruction": problem["instruction"],
        "output_path": str(output_path),
        "reasoning_path": str(reasoning_path) if reasoning_text is not None else None,
        "reasoning_text": reasoning_text,
        "extra_artifact_paths": [str(p) for p in extra_paths],
        "success": True,
        "skipped": True,
        "inference_time_s": None,
        "input_size_wh": input_size,
        "output_size_wh": output_size,
        "gpu_memory_before": {},
        "gpu_memory_after": {},
    }


def _mean(values: List[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None

def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    mid = len(s) // 2
    return round((s[mid] + s[~mid]) / 2, 4)


# ─── Cross-task scheduling ───────────────────────────────────────────────────
# A flat global plan + one shared semaphore (async) / ThreadPoolExecutor
# (sync) lets fast tasks unblock slow ones across task boundaries — instead
# of every task waiting at a per-task ``asyncio.gather`` barrier for its
# slowest problem before the next task can start. On runs with sparse
# work (resumes, partial cache, narrow ``--tasks`` filters) this is the
# difference between ~max(latency) wall and Σ max(latency_per_task) wall.

PlanEntry = Tuple[str, Dict, Path, bool]
"""Single entry in the global plan: (task_id, problem, output_path, is_cached)."""


def _build_global_plan(
    tasks: Dict[str, List[Dict]],
    max_problems: Optional[int],
    out_dir: Path,
    overwrite: bool,
) -> Tuple[List[PlanEntry], Dict[str, Tuple[int, int]]]:
    """Build a flat plan spanning all tasks plus a per-task summary.

    Returns ``(plan, summary)`` where plan is a list of
    ``(task_id, problem, output_path, is_cached)`` tuples and summary
    maps ``task_id -> (n_problems, n_cached)``. The cache check is a
    cheap ``Path.exists()``; per-problem decode validation happens
    later in :func:`_build_skipped_result` (forecast-cached files that
    fail to decode fall through and get redone).

    ``--tasks`` and ``--retry-failed`` filters are applied upstream
    in :func:`load_benchmark` / ``main()`` so the plan only sees
    problems that should run. ``--max-problems`` is applied here.
    """
    plan: List[PlanEntry] = []
    summary: Dict[str, Tuple[int, int]] = {}
    for task_id in sorted(tasks.keys()):
        problems = tasks[task_id]
        if max_problems:
            problems = problems[:max_problems]
        task_out_dir = out_dir / task_id.replace(".", "_")
        n_cached = 0
        for p in problems:
            output_path = task_out_dir / f"{p['index']:04d}_output.png"
            is_cached = (not overwrite) and output_path.exists()
            if is_cached:
                n_cached += 1
            plan.append((task_id, p, output_path, is_cached))
        summary[task_id] = (len(problems), n_cached)
    return plan, summary


def _print_plan(
    summary: Dict[str, Tuple[int, int]],
    parallelism_note: str,
) -> None:
    """Print the ``=== Plan ===`` block and the ``=== Running ===`` header.

    Plan replaces the per-task headers from the old per-task-gather flow:
    with global streaming, results from later tasks interleave with
    earlier ones in completion order, so per-task headers (``Task X
    (N problems)`` ... ``Task Y ...``) would be misleading.
    """
    print()
    print("=== Plan ===")
    total_problems = 0
    total_cached = 0
    tasks_with_work = 0
    name_width = max((len(tid) for tid in summary), default=0)
    for task_id in sorted(summary.keys()):
        n, cached = summary[task_id]
        total_problems += n
        total_cached += cached
        if n > cached:
            tasks_with_work += 1
        if cached == 0:
            note = f"{n} problems"
        elif cached == n:
            note = f"{n} problems — all cached"
        else:
            note = f"{n} problems, {cached} cached, running {n - cached}"
        print(f"  Task {task_id.ljust(name_width)}  ({note})")

    to_run = total_problems - total_cached
    print()
    if to_run > 0:
        print(
            f"  {to_run} problems to run across {tasks_with_work} tasks "
            f"(cached: {total_cached}/{total_problems})"
        )
    else:
        print(f"  All {total_problems} problems cached — nothing to run.")
    print()
    print(f"=== Running × {parallelism_note} ===")


async def _run_global_async(
    evaluator,
    plan: List[PlanEntry],
):
    """Async generator: yield ``(task_id, result_dict)`` per problem in
    ``plan``, in completion order (not problem-index order).

    All problems share a single ``asyncio.Semaphore(evaluator.concurrency)``
    — no per-task barrier. Cached problems bypass the semaphore entirely
    and resolve immediately. Caller updates ``task_results`` and totals
    via the on-result callback in ``main()``; keeping the bookkeeping
    there lets the heartbeat-save logic share state cleanly.
    """
    sem = asyncio.Semaphore(evaluator.concurrency)

    async def process_one(
        task_id: str, problem: Dict, output_path: Path, is_cached: bool,
    ) -> Tuple[str, Dict]:
        if is_cached:
            cached = _build_skipped_result(problem, output_path)
            if cached is not None:
                return task_id, cached
        async with sem:
            input_image = problem["input_image"]
            t_start = time.perf_counter()
            try:
                result = await evaluator.generate_async(
                    input_image, problem["instruction"]
                )
                inference_time = round(time.perf_counter() - t_start, 4)
                output_image, reasoning_text, extra_sidecars = (
                    _unpack_generate_result(result)
                )
                output_image = output_image.convert("RGB")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_image.save(output_path)
                reasoning_path = None
                if reasoning_text:
                    reasoning_path = output_path.with_suffix(".txt")
                    reasoning_path.write_text(reasoning_text, encoding="utf-8")
                extra_paths = _write_extra_sidecars(output_path, extra_sidecars)
                return task_id, {
                    "index": problem["index"],
                    "task": problem["task"],
                    "mode": problem["mode"],
                    "visual_condition": problem["visual_condition"],
                    "instruction": problem["instruction"],
                    "output_path": str(output_path),
                    "reasoning_path": str(reasoning_path) if reasoning_path else None,
                    "reasoning_text": reasoning_text,
                    "extra_artifact_paths": [str(p) for p in extra_paths],
                    "success": True,
                    "inference_time_s": inference_time,
                    "input_size_wh": list(input_image.size),
                    "output_size_wh": list(output_image.size),
                    "gpu_memory_before": {},
                    "gpu_memory_after": {},
                }
            except Exception as exc:
                return task_id, {
                    "index": problem["index"],
                    "task": problem["task"],
                    "mode": problem["mode"],
                    "visual_condition": problem["visual_condition"],
                    "success": False,
                    "error": str(exc),
                }

    coros = [process_one(*entry) for entry in plan]
    for coro in asyncio.as_completed(coros):
        yield await coro


def _finalize_task_results(task_results: Dict) -> None:
    """Populate per-task summary fields after streaming completes.

    With global streaming, results arrive in completion order; we sort
    each task's problem list by ``index`` so the metrics JSON is stable
    across runs (helps diffing). Then compute the per-task aggregates.

    Mirrors the field set in the global ``save_metrics()`` summary so
    the per-task and global blocks have symmetric stats (avg, median,
    min, max).
    """
    for info in task_results.values():
        info["problems"].sort(key=lambda r: r.get("index", 0))
        times = [
            r["inference_time_s"]
            for r in info["problems"]
            if r.get("inference_time_s") is not None
        ]
        info.update({
            "num_problems": len(info["problems"]),
            "num_successful": sum(1 for r in info["problems"] if r.get("success")),
            "num_skipped": sum(1 for r in info["problems"] if r.get("skipped")),
            "avg_inference_time_s": _mean(times),
            "median_inference_time_s": _median(times),
            "min_inference_time_s": round(min(times), 4) if times else None,
            "max_inference_time_s": round(max(times), 4) if times else None,
        })


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on PaintBench and save output images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    parser.add_argument("--model", required=True,
                        help="Model name (e.g. flux2-dev, qwen-image-edit, instruct-pix2pix, longcat-image-edit)")
    parser.add_argument("--benchmark", default="output",
                        help="Benchmark source. Either a local directory of "
                             "task subdirs (e.g. benchmarks/PaintBench) or an "
                             "HF dataset spec 'hf:<repo>' / 'hf:<repo>@<revision>' "
                             "(e.g. hf:PaintBench/PaintBench). HF mode requires "
                             "--benchmark-config to pick a dataset config.")
    parser.add_argument("--benchmark-config", default=None,
                        help="Config name within the benchmark source. Required "
                             "when --benchmark uses 'hf:' (e.g. PaintBench or "
                             "TinyGrafixBench). For local --benchmark this "
                             "overrides the friendly source name; defaults to "
                             "the directory basename.")
    parser.add_argument("--split", default="test",
                        help="Dataset split for HF benchmark sources "
                             "(e.g. 'test' or 'dev' for PaintBench). Ignored "
                             "for local --benchmark paths.")
    parser.add_argument("--out-dir", default="model_outputs",
                        help="Directory to save output images and metrics JSON")
    parser.add_argument("--api-key",
                        help="API key for API-based models")
    parser.add_argument("--tasks",
                        help="Comma-separated task IDs to run (e.g. 1.1,3.1,4.2). Default: all.")
    parser.add_argument("--max-problems", type=int,
                        help="Max problems per task (default: all)")
    parser.add_argument("--retry-failed", type=Path, default=None,
                        help="Path to a metrics JSON from a prior run; re-run "
                             "only the problems that failed there. Output goes "
                             "into the same --out-dir, overwriting any stale "
                             "PNGs and writing a fresh metrics_*.json for the "
                             "retry pass.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Redo problems whose output PNG already exists, "
                             "overwriting the cached result. By default "
                             "inference is incremental: existing PNGs are "
                             "reused after a Pillow .load() decode check, "
                             "so reruns after a cancel / crash only redo "
                             "the missing problems. Pass --overwrite when "
                             "you change inference logic (prompt template, "
                             "sampling config, gateway routing, ...) and "
                             "want to invalidate the cache without manually "
                             "deleting --out-dir. Composes with "
                             "--retry-failed (which filters problems) — "
                             "--overwrite controls cache reuse.")

    # Hardware
    parser.add_argument("--device",
                        help="GPU device to use (e.g. cuda, cuda:0, cpu). Auto-detected if omitted.")
    parser.add_argument("--device-map",
                        help="Accelerate device_map strategy for multi-GPU (e.g. balanced)")
    parser.add_argument("--max-memory-gb-per-gpu", type=int,
                        help="Max GPU memory in GB per device (used with --device-map)")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"],
                        default="bfloat16")

    # Generation — all default to None so each pipeline's own defaults apply
    # when not explicitly overridden on the command line.
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Number of denoising steps. Uses each pipeline's default if omitted.")
    parser.add_argument("--guidance-scale", type=float, default=None,
                        help="CFG guidance scale. Uses each pipeline's default if omitted.")
    parser.add_argument("--image-guidance-scale", type=float, default=None,
                        help="Image guidance scale (instruct-pix2pix only).")
    parser.add_argument("--negative-prompt", default=None,
                        help="Negative prompt (LongCat, Qwen).")
    parser.add_argument("--true-cfg-scale", type=float, default=None,
                        help="True CFG scale (Qwen only).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")

    # BAGEL-specific
    parser.add_argument("--bagel-repo", default=None,
                        help="Path to an existing BAGEL upstream clone (BAGEL model only). "
                             "Use to point at an offline / air-gapped checkout. Falls back "
                             "to $BAGEL_REPO env var, then to an auto-cloned upstream pinned "
                             "in src/inference.py at $HF_HUB_CACHE/paintbench/bagel-upstream.")
    parser.add_argument("--no-think", action="store_true",
                        help="Disable chain-of-thought reasoning for BAGEL (think=True by default).")

    # Hunyuan-specific overrides — `bot_task` controls whether the model does
    # CoT thinking + recaption before image generation. `use_system_prompt`
    # selects which system prompt template is used. Defaults are set in
    # HunyuanImage3InstructDistilModel.__init__; CLI overrides take precedence.
    parser.add_argument("--bot-task", default=None,
                        choices=["image", "auto", "recaption", "think", "think_recaption", "img_ratio"],
                        help="Hunyuan only. `image`=direct gen (no CoT/rewrite); "
                             "`recaption`=rewrite→image; `think`=think→image; "
                             "`think_recaption`=think→rewrite→image (recommended for editing).")
    parser.add_argument("--use-system-prompt", default=None,
                        choices=["None", "en_vanilla", "en_recaption", "en_think_recaption",
                                 "en_unified", "dynamic", "custom"],
                        help="Hunyuan only. System prompt template. "
                             "`None`=no system prompt; `en_vanilla`=plain T2I; "
                             "`en_unified`=full T2I+TI2I editing protocol (default).")
    parser.add_argument("--model-cache-dir", default=None,
                        help="Hunyuan only. Override the download root for the "
                             "sanitized (dotless) snapshot. Default is "
                             "<HF cache>/paintbench/<variant>, resolved from "
                             "HF_HUB_CACHE / HF_HOME like huggingface_hub. "
                             "Precedence: this flag > $HUNYUAN_MODEL_DIR > default.")
    parser.add_argument("--high-vram", action="store_true",
                        help="LongCat only. Skip enable_model_cpu_offload() and "
                             "load the full pipeline on GPU. The default offload "
                             "path breaks on Qwen2.5-VL's attention proj layer "
                             "(accelerate hooks miss it -> mat1/weights "
                             "device-mismatch RuntimeError). Use on any GPU with "
                             "≥24 GB VRAM (LongCat is ~18 GB on-device).")

    # Gemini-specific overrides — precedence is CLI > env var > class default.
    # Useful for running variants (e.g. an older Gemini Flash variant alongside
    # the default) without reshuffling .env between runs.
    parser.add_argument("--model-name", default=None,
                        help="Override the underlying model identifier (Gemini models). "
                             "Falls back to GEMINI_MODEL_NAME env var, then class default.")
    parser.add_argument("--base-url", default=None,
                        help="Override the SDK base URL (Gemini models). "
                             "Falls back to GEMINI_BASE_URL env var, then SDK default.")
    parser.add_argument("--include-thoughts", choices=["0", "1"], default=None,
                        help="Override include_thoughts (Gemini models). "
                             "Falls back to GEMINI_INCLUDE_THOUGHTS env var, then class default.")

    # Inference concurrency — N concurrent calls per task. Default None means
    # "use the path's default": sync ThreadPoolExecutor falls back to 1
    # (sequential), async-capable models (NanoBanana*, GptImage2) fall back to
    # their class-level concurrency default. Setting --workers explicitly caps
    # both paths uniformly.
    parser.add_argument("--workers", type=int, default=None,
                        help="Concurrent model calls per task. Sync (threaded) "
                             "and async paths both honour this. Unset = sync "
                             "runs serial, async uses the model's class default. "
                             "4–8 is a reasonable explicit setting for remote "
                             "API models.")

    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be >= 1")

    # Dispatch the benchmark source. Local paths get a LocalBenchmarkSource;
    # 'hf:<repo>[@<rev>]' specs get an HfBenchmarkSource (requires
    # --benchmark-config). ``out_dir`` is computed AFTER the evaluator
    # loads — the subdir name (``output_dir_slug``) can depend on the
    # resolved model variant. Defer until then.
    try:
        benchmark_source = parse_benchmark_arg(
            args.benchmark,
            config=args.benchmark_config,
            split=args.split,
        )
    except (ValueError, FileNotFoundError, NotADirectoryError) as exc:
        parser.error(str(exc))

    task_filter = [t.strip() for t in args.tasks.split(",")] if args.tasks else None

    # ── System info ───────────────────────────────────────────────────────────
    system_info = get_system_info()

    print("\n=== System Info ===")
    print(f"  Platform  : {system_info['platform']}")
    print(f"  Python    : {system_info['python_version']}")
    print(f"  PyTorch   : {system_info['torch_version']}")
    print(f"  CUDA      : {system_info['cuda_version']}")
    print(f"  Num GPUs  : {system_info['num_gpus']}")
    for gpu in system_info["gpus"]:
        print(f"    GPU {gpu['index']}: {gpu['name']}  ({gpu['total_memory_gb']} GB VRAM, "
              f"CC {gpu['compute_capability']}, {gpu['multi_processor_count']} SMs)")
    print(f"  CPU       : {system_info['cpu_model']} ({system_info['logical_cpu_cores']} logical cores)")
    if system_info["ram_total_gb"]:
        print(f"  RAM       : {system_info['ram_total_gb']} GB")

    # ── Load benchmark ────────────────────────────────────────────────────────
    print(f"\n=== Loading Benchmark: {benchmark_source.name()} ===")
    print(f"  Revision : {benchmark_source.revision()}")
    tasks = load_benchmark(benchmark_source, task_filter)

    if args.retry_failed:
        prior = json.loads(args.retry_failed.read_text())
        failed_keys = _failed_problem_keys(prior)
        if not failed_keys:
            print(f"\n  No failures in {args.retry_failed} — nothing to retry.")
            return
        # Filter to only the problems whose prior run was unsuccessful.
        tasks = {
            tid: [p for p in plist if (tid, p["index"]) in failed_keys]
            for tid, plist in tasks.items()
        }
        tasks = {tid: plist for tid, plist in tasks.items() if plist}
        print(f"\n  Retry mode: {sum(len(p) for p in tasks.values())} failed problems "
              f"across {len(tasks)} tasks (from {args.retry_failed.name})")

    total_problems_count = sum(
        min(len(p), args.max_problems) if args.max_problems else len(p)
        for p in tasks.values()
    )
    print(f"  Tasks    : {len(tasks)}")
    print(f"  Problems : {total_problems_count}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\n=== Loading Model: {args.model} ===")

    # Only forward args that were explicitly set; None means "use model default".
    model_kwargs = {k: v for k, v in dict(
        device=args.device,
        device_map=args.device_map,
        max_memory_gb_per_gpu=args.max_memory_gb_per_gpu,
        torch_dtype=args.torch_dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        negative_prompt=args.negative_prompt,
        true_cfg_scale=args.true_cfg_scale,
        seed=args.seed,
        bagel_repo=args.bagel_repo,
    ).items() if v is not None}
    # --no-think disables CoT for BAGEL; always include so default (True) is explicit.
    model_kwargs["think"] = not args.no_think

    # Hunyuan overrides — only forward when the CLI flag was set. Every
    # model __init__ in the registry accepts ``**_`` for unrecognised
    # kwargs, so even if the user pairs ``--bot-task`` with a non-Hunyuan
    # model (e.g. ``--model flux2-dev --bot-task think``) the kwarg is
    # silently dropped rather than crashing — consistent with how the
    # Gemini overrides above behave. A future "warn on model-specific
    # flag with non-matching model" pass would close this hole.
    if args.bot_task is not None:
        model_kwargs["bot_task"] = args.bot_task
    if args.use_system_prompt is not None:
        # CLI string "None" → Python None for the get_system_prompt branch
        model_kwargs["use_system_prompt"] = (
            None if args.use_system_prompt == "None" else args.use_system_prompt
        )
    if args.model_cache_dir is not None:
        model_kwargs["model_cache_dir"] = args.model_cache_dir
    if args.high_vram:
        model_kwargs["high_vram"] = True

    # Gemini overrides: forward only when the user explicitly set them on the
    # CLI; None means "let the constructor's env/default lookup take over".
    if args.model_name is not None:
        model_kwargs["model_name"] = args.model_name
    if args.base_url is not None:
        model_kwargs["base_url"] = args.base_url
    if args.include_thoughts is not None:
        model_kwargs["include_thoughts"] = args.include_thoughts == "1"

    # When --workers is explicitly set, cap the API model's async concurrency
    # to match. NanoBanana1/2 and GptImage2 all accept ``concurrency`` and use
    # an asyncio.Semaphore in _run_tasks_async; sync models ignore the unknown
    # kwarg via their **_ catch-all.
    if args.workers is not None:
        model_kwargs["concurrency"] = args.workers

    t_load_start = time.perf_counter()
    evaluator = create_evaluator(args.model, api_key=args.api_key, **model_kwargs)
    evaluator.load_model()
    model_load_time_s = round(time.perf_counter() - t_load_start, 2)
    model_info = evaluator.get_model_info()

    print(f"  Model loaded in {model_load_time_s}s")

    gpu_mem_after_load = get_gpu_memory_snapshot()
    for dev, mem in gpu_mem_after_load.items():
        print(f"  {dev} after load: {mem['allocated_gb']} GB allocated, {mem['reserved_gb']} GB reserved")

    # Resolve output directory now that the evaluator's model variant is
    # known. ``output_dir_slug`` defaults to the registry key but a model
    # that proxies to multiple underlying backends can override it to
    # encode the variant id so different backends don't clobber each
    # other's outputs.
    output_subdir = evaluator.output_dir_slug(args.model)
    out_dir = Path(args.out_dir) / output_subdir / benchmark_source.name()
    out_dir.mkdir(parents=True, exist_ok=True)
    if output_subdir != args.model:
        print(
            f"  Output subdir: {output_subdir}/  (use this as MODEL=... "
            f"for downstream make eval / stats / report)"
        )

    # ── Build the global plan ─────────────────────────────────────────────────
    plan, plan_summary = _build_global_plan(
        tasks, args.max_problems, out_dir, args.overwrite,
    )
    forecast_cached = sum(c for _, c in plan_summary.values())

    # ── Run inference ─────────────────────────────────────────────────────────
    is_async = hasattr(evaluator, "generate_async")
    # Always defined so the sync-path block below can reference it without
    # tripping the "potentially unbound" lint (used only when not is_async).
    sync_workers = args.workers if args.workers is not None else 1
    if is_async:
        parallelism_note = f"async {evaluator.concurrency}"
    else:
        parallelism_note = (
            f"{sync_workers} workers" if sync_workers > 1 else "serial"
        )
    _print_plan(plan_summary, parallelism_note)

    run_start = datetime.now()
    task_results: Dict = {
        task_id: {"task_id": task_id, "problems": []}
        for task_id in plan_summary
    }
    total_attempted = 0
    total_successful = 0
    total_skipped = 0

    timestamp = run_start.strftime("%Y%m%d_%H%M%S")
    metrics_path = out_dir / f"inference_metrics_{timestamp}.json"

    # Stable fields that don't change between images.
    metrics_base = {
        "model_name": args.model,
        "run_timestamp": run_start.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": system_info,
        "model_info": model_info,
        "model_load": {
            "load_time_s": model_load_time_s,
            "gpu_memory_after_load": gpu_mem_after_load,
        },
    }

    def save_metrics() -> None:
        """Recompute summary stats from current state and write metrics JSON."""
        all_times = [
            p["inference_time_s"]
            for t in task_results.values()
            for p in t["problems"]
            if p.get("inference_time_s") is not None
        ]
        wall_time_s = round((datetime.now() - run_start).total_seconds(), 2)
        # Throughput / new-run success rate are meaningful only for problems
        # we actually ran, since skipped (cached) problems contribute zero
        # wall time and trivially "succeed" via the cache. Reporting them
        # over (successful - skipped) keeps the numbers interpretable on
        # resumed runs and avoids the surprise where a 100% cached resume
        # shows throughput=0.0 despite zero failures.
        new_run_attempted = total_attempted - total_skipped
        new_run_successful = total_successful - total_skipped
        throughput = (round(new_run_successful / (wall_time_s / 60), 2)
                      if wall_time_s > 0 and new_run_successful > 0 else None)
        new_run_success_rate = (round(new_run_successful / new_run_attempted, 4)
                                if new_run_attempted else None)
        metrics = {
            **metrics_base,
            "summary": {
                "total_tasks": len(task_results),
                "total_problems_attempted": total_attempted,
                "total_problems_successful": total_successful,
                "total_problems_skipped": total_skipped,
                # success_rate is over ALL problems (cached + newly run); it
                # measures "fraction of problems with a usable output". For
                # the rate of newly-run problems that succeeded — i.e.
                # "did the model actually work this run" — see
                # new_run_success_rate.
                "success_rate": round(total_successful / total_attempted, 4) if total_attempted else 0.0,
                "new_run_success_rate": new_run_success_rate,
                "total_wall_time_s": wall_time_s,
                "avg_inference_time_s": _mean(all_times),
                "median_inference_time_s": _median(all_times),
                "min_inference_time_s": round(min(all_times), 4) if all_times else None,
                "max_inference_time_s": round(max(all_times), 4) if all_times else None,
                "throughput_problems_per_min": throughput,
            },
            "tasks": task_results,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2))

    # Heartbeat: save metrics every ~5s of wall time (instead of after every
    # result). With high concurrency / many fast cached returns the after-
    # every-result variant thrashes the JSON.
    last_save_time = time.monotonic()

    def on_result(task_id: str, result: Dict) -> None:
        nonlocal total_attempted, total_successful, total_skipped, last_save_time

        task_results[task_id]["problems"].append(result)
        total_attempted += 1

        idx = result.get("index", -1)
        if result.get("skipped"):
            total_successful += 1
            total_skipped += 1
            # Per-problem print suppressed for cached — Plan reports the
            # count up front.
        elif result.get("success"):
            total_successful += 1
            t = result.get("inference_time_s") or 0.0
            print(f"  [{task_id} {idx:04d}]  {t:.2f}s  OK")
        else:
            err = result.get("error", "unknown error")
            print(f"  [{task_id} {idx:04d}]  FAILED — {err}")

        now = time.monotonic()
        if now - last_save_time >= 5.0:
            save_metrics()
            last_save_time = now

    if is_async:
        async def _drive_async() -> None:
            async for task_id, result in _run_global_async(evaluator, plan):
                on_result(task_id, result)
        asyncio.run(_drive_async())
    else:
        # Sync path: one ThreadPoolExecutor across ALL tasks (vs the old
        # per-task pool). Threads, not processes — agent CLI / API calls
        # are I/O-bound and clients are thread-safe, and threads sidestep
        # the macOS sandbox restriction on POSIX semaphores that
        # multiprocessing trips (see ``JOBS=1`` workaround in CLAUDE.md).

        def _run_one_sync(
            task_id: str, problem: Dict, output_path: Path, is_cached: bool,
        ) -> Tuple[str, Dict]:
            if is_cached:
                cached = _build_skipped_result(problem, output_path)
                if cached is not None:
                    return task_id, cached
            try:
                return task_id, run_single(evaluator, problem, output_path)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return task_id, {
                    "index": problem["index"],
                    "task": problem["task"],
                    "mode": problem["mode"],
                    "visual_condition": problem["visual_condition"],
                    "success": False,
                    "error": str(exc),
                }

        with ThreadPoolExecutor(max_workers=sync_workers) as ex:
            futures = [
                ex.submit(_run_one_sync, *entry) for entry in plan
            ]
            for future in as_completed(futures):
                task_id, result = future.result()
                on_result(task_id, result)

    _finalize_task_results(task_results)
    _print_corrupt_cache_note(forecast_cached, total_skipped)

    # ── Final summary ─────────────────────────────────────────────────────────
    save_metrics()  # final write with complete per-task summaries
    final = json.loads(metrics_path.read_text())["summary"]
    print("\n=== Done ===")
    print(f"  Output dir   : {out_dir}")
    print(f"  Metrics JSON : {metrics_path.name}")
    print(f"  Successful   : {final['total_problems_successful']} / {final['total_problems_attempted']}")
    if final.get("total_problems_skipped"):
        ran = final["total_problems_successful"] - final["total_problems_skipped"]
        print(f"  Skipped      : {final['total_problems_skipped']} (cached); {ran} actually run")
    print(f"  Wall time    : {final['total_wall_time_s']}s")
    print(f"  Avg latency  : {final['avg_inference_time_s']}s / problem")
    print(f"  Throughput   : {final['throughput_problems_per_min']} problems/min")


if __name__ == "__main__":
    main()
