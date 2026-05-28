"""Pluggable benchmark sources for the inference pipeline.

This module abstracts over two storage backends for benchmark problems:

* :class:`LocalBenchmarkSource` — the original on-disk layout
  ``<root>/<task>/<NNN>_input.png + <NNN>_answer.png + <NNN>.json``.
  Used by every existing ``--benchmark <path>`` invocation; behaviour is
  preserved bit-for-bit.

* :class:`HfBenchmarkSource` — loads via
  ``datasets.load_dataset(repo_id, config, split=..., revision=...)``.
  Lets ``--benchmark hf:PaintBench/PaintBench`` work without a local
  clone, with a pinnable git ref for paper-grade reproducibility.

Both yield the same :class:`Problem` shape. :func:`parse_benchmark_arg`
is the dispatcher used by ``src/inference.py``'s argparse layer.

Scope: this module is only consumed by ``src/inference.py`` in Phase 1a.
``eval.py``'s ``--benchmarks <all-root>`` boundary iterates subdirs
across multiple configs and needs its own design pass (Phase 1b).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from PIL import Image


# ─── Problem ────────────────────────────────────────────────────────────────

@dataclass
class Problem:
    """One benchmark problem, agnostic of where its images live.

    The ``_input`` / ``_answer`` fields hold either a filesystem
    :class:`pathlib.Path` (local source — decoded lazily on access via the
    :pyattr:`input_image` / :pyattr:`answer_image` properties) or a
    :class:`PIL.Image.Image` already resident in memory (HF source — the
    ``datasets`` library decodes Image-feature columns on row access).

    The class supports both attribute-style access (``problem.input_image``)
    and dict-style access (``problem["input_image"]``) so the existing
    ``src/inference.py`` orchestrator — which treats problems as dicts —
    keeps working without a wholesale refactor.
    """

    pid: int
    task: str
    mode: str
    visual_condition: str
    instruction: str
    metadata: dict = field(default_factory=dict)
    _input: Union[Path, Image.Image] = None  # type: ignore[assignment]
    _answer: Optional[Union[Path, Image.Image]] = None

    # ── Lazy image accessors ──────────────────────────────────────────────

    @property
    def input_image(self) -> Image.Image:
        """Decoded RGB input image. Opens the file each call for Path
        sources; returns the in-memory image for HF sources."""
        return _to_rgb(self._input)

    @property
    def answer_image(self) -> Optional[Image.Image]:
        """Decoded RGB answer image, or ``None`` for problems without one."""
        if self._answer is None:
            return None
        return _to_rgb(self._answer)

    @property
    def input_size_wh(self) -> tuple[int, int]:
        """``(width, height)`` of the input image — cheap.

        Prefers the JSON metadata's ``W``/``H`` so the cache-check fast
        path in ``inference._build_skipped_result`` doesn't force a PNG
        decode for every cached problem. Falls back to a Pillow
        header-read for Path sources or ``.size`` for in-memory ones.
        """
        if "W" in self.metadata and "H" in self.metadata:
            try:
                return int(self.metadata["W"]), int(self.metadata["H"])
            except (TypeError, ValueError):
                pass
        if isinstance(self._input, Path):
            with Image.open(self._input) as img:
                return img.size
        return self._input.size  # already-decoded PIL.Image

    # ── Dict-style access ────────────────────────────────────────────────
    # Lets the inference.py orchestrator keep using ``problem["..."]`` to
    # access fields (it was previously a plain dict). The legacy key
    # ``index`` maps to :attr:`pid` for backwards compat with prior
    # metrics-JSON consumers.

    _KEY_MAP = {
        "pid":              "pid",
        "index":            "pid",  # legacy alias
        "task":             "task",
        "mode":             "mode",
        "visual_condition": "visual_condition",
        "instruction":      "instruction",
        "metadata":         "metadata",
        "input_image":      "input_image",
        "answer_image":     "answer_image",
        "input_size_wh":    "input_size_wh",
    }

    def __getitem__(self, key: str) -> Any:
        try:
            attr = self._KEY_MAP[key]
        except KeyError as exc:
            raise KeyError(key) from exc
        return getattr(self, attr)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return key in self._KEY_MAP


def _to_rgb(src: Union[Path, Image.Image]) -> Image.Image:
    """Materialise ``src`` as a decoded RGB :class:`PIL.Image.Image`."""
    if isinstance(src, Path):
        return Image.open(src).convert("RGB")
    # In-memory image (e.g. from the HF ``datasets`` library). Avoid a
    # round-trip when already in the right mode — ``.convert("RGB")``
    # would otherwise copy unnecessarily.
    return src if src.mode == "RGB" else src.convert("RGB")


# ─── BenchmarkSource interface ──────────────────────────────────────────────

class BenchmarkSource(ABC):
    """Iterates problems for a benchmark config; hides local-vs-HF storage."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable benchmark name (e.g. ``"PaintBench"``). Printed
        in the inference banner and used as the output-dir subfolder."""

    @abstractmethod
    def revision(self) -> str:
        """Reproducibility fingerprint. For local sources this is the
        directory path; for HF sources it's ``hf:<repo>@<sha> split=...``."""

    @abstractmethod
    def iter_tasks(self) -> Iterator[str]:
        """Yield task names in sorted (deterministic) order."""

    @abstractmethod
    def iter_problems(self, task: str) -> Iterator[Problem]:
        """Yield :class:`Problem` instances for ``task`` in pid order."""


# ─── LocalBenchmarkSource ───────────────────────────────────────────────────

class LocalBenchmarkSource(BenchmarkSource):
    """Reads problems from the canonical on-disk layout::

        <root>/<task>/<NNN>_input.png
        <root>/<task>/<NNN>_answer.png
        <root>/<task>/<NNN>.json        ← per-problem metadata

    The JSON sidecar carries the rich fields (``task``, ``mode``,
    ``visual_condition``, ``instruction`` and all generator params); only
    its values are forwarded into the :class:`Problem`. Image paths are
    held lazily; nothing is decoded until the consumer reads
    ``problem.input_image`` / ``problem.answer_image``.

    Behaviour matches the pre-refactor ``inference.load_benchmark`` for
    the existing Layout B (``<task>/<NNN>_*``) used by every current
    benchmark. The legacy Layout A (``task_*/problem_NNNN_*``) is no
    longer in use and is intentionally not re-implemented here.
    """

    def __init__(self, root: Path, name: str):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Benchmark dir not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Benchmark dir is not a directory: {root}")
        self.root = root
        self._name = name

    def name(self) -> str:
        return self._name

    def revision(self) -> str:
        return str(self.root)

    def iter_tasks(self) -> Iterator[str]:
        for child in sorted(self.root.iterdir()):
            if child.is_dir():
                yield child.name

    def iter_problems(self, task: str) -> Iterator[Problem]:
        task_dir = self.root / task
        if not task_dir.is_dir():
            return
        for input_path in sorted(task_dir.glob("*_input.png")):
            stem = input_path.name.removesuffix("_input.png")
            json_path = task_dir / f"{stem}.json"
            if not json_path.exists():
                # Mirror the pre-refactor behaviour: silently drop
                # problems missing their metadata sidecar.
                continue
            try:
                meta = json.loads(json_path.read_text())
            except json.JSONDecodeError:
                continue
            answer_path = task_dir / f"{stem}_answer.png"
            pid = int(meta.get("problem_id", stem))
            yield Problem(
                pid=pid,
                task=meta.get("task", task),
                mode=meta.get("mode") or "",
                visual_condition=meta.get("visual_condition") or "",
                instruction=meta["instruction"],
                metadata=meta,
                _input=input_path,
                _answer=answer_path if answer_path.exists() else None,
            )


# ─── HfBenchmarkSource ──────────────────────────────────────────────────────

class HfBenchmarkSource(BenchmarkSource):
    """Loads problems from a Hugging Face dataset repo.

    Wraps ``datasets.load_dataset(repo_id, config, split=..., revision=...)``.
    Pinning ``revision`` to a commit SHA gives bit-for-bit reproducible
    benchmark loads across runs — the public-release UX target.

    Memory: the underlying ``datasets`` library decodes Image-feature
    columns on row access; :meth:`iter_problems` materialises each
    Problem's PIL images into memory, so a full ``test`` split for
    PaintBench (2016 rows × ~2 MB/image × 2 images) sits at a few GB
    while inference runs. Use ``--tasks`` to subset for memory-bound
    environments, or use the ``dev`` split (280 rows) for quick checks.
    """

    def __init__(
        self,
        repo_id: str,
        config: str,
        *,
        split: str = "test",
        revision: Optional[str] = None,
    ):
        # Local import so the rest of the inference pipeline (sync /
        # offline / no-extras installs) doesn't require the `data` extra.
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "HfBenchmarkSource requires the `datasets` package. "
                "Install via `uv sync --extra data` or `pip install datasets`."
            ) from exc

        self.repo_id = repo_id
        self.config = config
        self.split = split
        self._revision_spec = revision

        self._dataset = load_dataset(
            repo_id, config, split=split, revision=revision,
        )

        # Build a task → list[row_idx] map by reading the cheap string
        # columns only. Avoids decoding any images at construction time.
        task_col = self._dataset["task"]
        pid_col = self._dataset["problem_id"]
        self._task_to_indices: dict[str, list[int]] = {}
        for i, task in enumerate(task_col):
            self._task_to_indices.setdefault(task, []).append(i)
        # Stable pid-ordered iteration within each task.
        for indices in self._task_to_indices.values():
            indices.sort(key=lambda i: int(pid_col[i]))

        # Resolve the SHA for the banner. Best-effort — falls back to the
        # user-supplied spec (or ``"main"``) if the API call is blocked
        # (e.g. cache-only mode).
        self._resolved_sha: Optional[str] = self._resolve_sha(repo_id, revision)

    @staticmethod
    def _resolve_sha(repo_id: str, revision: Optional[str]) -> Optional[str]:
        try:
            from huggingface_hub import HfApi

            info = HfApi().dataset_info(repo_id, revision=revision)
            return info.sha
        except Exception:
            return None

    def name(self) -> str:
        return self.config

    def revision(self) -> str:
        sha = self._resolved_sha or self._revision_spec or "main"
        return f"hf:{self.repo_id}@{sha} split={self.split}"

    def iter_tasks(self) -> Iterator[str]:
        yield from sorted(self._task_to_indices)

    def iter_problems(self, task: str) -> Iterator[Problem]:
        ds = self._dataset
        for idx in self._task_to_indices.get(task, []):
            row = ds[idx]
            meta_raw = row.get("metadata")
            if isinstance(meta_raw, str):
                try:
                    meta = json.loads(meta_raw)
                except json.JSONDecodeError:
                    meta = {}
            elif isinstance(meta_raw, dict):
                meta = meta_raw
            else:
                meta = {}
            yield Problem(
                pid=int(row["problem_id"]),
                task=row["task"],
                mode=row.get("mode") or "",
                visual_condition=row.get("visual_condition") or "",
                instruction=row["instruction"],
                metadata=meta,
                _input=row["input_image"],
                _answer=row.get("answer_image"),
            )


# ─── CLI dispatcher ─────────────────────────────────────────────────────────

def parse_benchmark_arg(
    spec: str,
    *,
    config: Optional[str] = None,
    split: str = "test",
) -> BenchmarkSource:
    """Build a :class:`BenchmarkSource` from a ``--benchmark`` CLI value.

    Dispatch rules:

    * ``hf:<repo>`` or ``hf:<repo>@<revision>`` → :class:`HfBenchmarkSource`.
      ``config`` is required (the upload script publishes ``PaintBench``
      and ``TinyGrafixBench`` as separate configs); ``split`` defaults to
      ``"test"``.

    * anything else → :class:`LocalBenchmarkSource` pointed at the path.
      ``config`` overrides the friendly :py:meth:`BenchmarkSource.name`;
      it defaults to the directory basename.
    """
    if spec.startswith("hf:"):
        rest = spec[len("hf:"):]
        if "@" in rest:
            repo_id, revision = rest.split("@", 1)
            if not revision:
                raise ValueError(
                    f"Malformed --benchmark spec {spec!r}: empty revision "
                    "after '@' (use hf:<repo> for no revision pin)."
                )
        else:
            repo_id, revision = rest, None
        if not repo_id:
            raise ValueError(
                f"Malformed --benchmark spec {spec!r}: expected hf:<repo> "
                "or hf:<repo>@<revision>."
            )
        if not config:
            raise ValueError(
                "--benchmark-config is required when --benchmark uses an "
                "'hf:' source (e.g. --benchmark-config PaintBench)."
            )
        return HfBenchmarkSource(repo_id, config, split=split, revision=revision)

    path = Path(spec)
    return LocalBenchmarkSource(path, name=config or path.name)
