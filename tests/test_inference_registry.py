"""Inference model registry sanity. No weights loaded, no GPU needed —
just verifies the registry hasn't drifted (renamed class, missing entry,
forgotten BaseModel inheritance, etc.). Catches the kind of regression
where someone refactors a model class and silently breaks `--model X`."""
from __future__ import annotations

import inspect
import subprocess
import threading
from pathlib import Path

import pytest
from PIL import Image

import inference

EXPECTED_MODELS = {
    "instruct-pix2pix",
    "longcat-image-edit",
    "qwen-image-edit",
    "flux2-dev",
    "flux1-kontext-dev",
    "flux2-klein-9b",
    "bagel",
    "nano-banana-1",
    "nano-banana-2",
    "gpt-image-2",
    "hunyuan-image-3",
    "hunyuan-image-3-instruct",
}


def test_registry_has_expected_models():
    assert set(inference._REGISTRY) == EXPECTED_MODELS, \
        f"Registry drift. Expected {EXPECTED_MODELS}, got {set(inference._REGISTRY)}"


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
def test_registry_entries_are_basemodel_subclasses(name):
    cls = inference._REGISTRY[name]
    assert inspect.isclass(cls), f"{name!r} → {cls!r} is not a class"
    assert issubclass(cls, inference.BaseModel), \
        f"{cls.__name__} doesn't subclass BaseModel"


@pytest.mark.parametrize("name", sorted(EXPECTED_MODELS))
def test_registry_entries_override_required_methods(name):
    """Both load_model and generate must be implemented (directly or via
    inheritance from another model class) — if a subclass forgets one, calling
    it would hit BaseModel's NotImplementedError only at runtime."""
    cls = inference._REGISTRY[name]
    for method in ("load_model", "generate"):
        # Walk the MRO to find where the method is actually defined.
        # Reject if it's still BaseModel (which raises NotImplementedError).
        defined_on = next(
            (c for c in cls.__mro__ if method in c.__dict__),
            None,
        )
        assert defined_on is not None and defined_on is not inference.BaseModel, \
            f"{cls.__name__}.{method} is not implemented (still inherits from BaseModel)"


def test_create_evaluator_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        inference.create_evaluator("not-a-real-model")


def test_create_evaluator_is_case_insensitive():
    """The CLI uses model names verbatim; verify a casing slip still works
    (create_evaluator lowercases internally per docstring). Use nano-banana-2
    because the local diffusion model __init__s call torch.cuda.is_available(),
    which fails without the [inference] extra installed (CI runs core-only)."""
    # Construct with a placeholder api_key so __init__'s key check passes.
    # No load_model() — that would actually hit the Gemini API.
    evaluator = inference.create_evaluator("NANO-BANANA-2", api_key="test")
    assert isinstance(evaluator, inference.BaseModel)


def test_nano_banana_requires_api_key():
    """API model should fail fast in __init__ if no key is provided.
    Exercises the only model whose __init__ can raise."""
    import os
    saved_keys = {k: os.environ.pop(k, None) for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
    try:
        with pytest.raises(ValueError, match="API key required"):
            inference.create_evaluator("nano-banana-2")
    finally:
        for k, v in saved_keys.items():
            if v is not None:
                os.environ[k] = v


def test_hunyuan_forwards_input_size_to_pipeline():
    class FakePipe:
        def generate_image(self, **kwargs):
            self.kwargs = kwargs
            return [""], [Image.new("RGB", (17, 23), "white")]

    pipe = FakePipe()
    model = inference.HunyuanImage3InstructDistilModel.__new__(
        inference.HunyuanImage3InstructDistilModel,
    )
    model.pipe = pipe
    model.seed = 123
    model.use_system_prompt = False
    model.bot_task = "image-editing"
    model.diff_infer_steps = 8
    # ``generate()`` serializes the pipeline call on ``self._lock`` (see
    # BaseModel thread-safety contract). Bypassing ``__init__`` via
    # ``__new__`` skips the lock init, so seed it manually here.
    model._lock = threading.Lock()

    image = Image.new("RGB", (17, 23), "black")
    output = model.generate(image, "recolor the shape")

    assert output.size == (17, 23)
    assert pipe.kwargs["image_size"] == "23x17"


# ── Hunyuan default cache-dir resolution ──────────────────────────────────────
# These exercise the priority chain documented on
# HunyuanImage3InstructDistilModel._default_model_cache_dir():
#   HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > $HF_HOME/hub
#       > $XDG_CACHE_HOME/huggingface/hub > ~/.cache/huggingface/hub
# Calling the classmethod directly avoids __init__ side-effects and never
# touches GPU / weights.

_HUNYUAN_HF_ENV_VARS = (
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HOME",
    # XDG_CACHE_HOME isn't HF-specific, but huggingface_hub honours it as
    # its next-to-last fallback when no HF cache var is set; strip it so
    # tests that exercise the $HOME-level fallback aren't surprised by an
    # XDG redirect leaking in from the host env (CI, devcontainers, …).
    "XDG_CACHE_HOME",
)


def _isolated_hf_env(monkeypatch):
    """Strip all HF cache env vars so each test starts from a known
    blank slate. Tests then set only the var(s) they care about."""
    for var in _HUNYUAN_HF_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize(
    "model_cls,expected_dirname",
    [
        (inference.HunyuanImage3InstructDistilModel, "HunyuanImage-3-Instruct-Distil"),
        (inference.HunyuanImage3InstructModel, "HunyuanImage-3-Instruct"),
    ],
)
def test_hunyuan_default_cache_dir_uses_hf_hub_cache(
    monkeypatch, tmp_path, model_cls, expected_dirname
):
    """HF_HUB_CACHE wins outright — both Hunyuan variants land under it,
    each in its own ``paintbench/<variant>/`` sub-subdir."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    resolved = model_cls._default_model_cache_dir()

    assert resolved == str(tmp_path / "paintbench" / expected_dirname)


def test_hunyuan_default_cache_dir_hf_hub_cache_wins_over_hf_home(monkeypatch, tmp_path):
    """If both are set, HF_HUB_CACHE takes precedence over HF_HOME — the
    documented priority chain."""
    _isolated_hf_env(monkeypatch)
    hub_dir = tmp_path / "hub_cache"
    home_dir = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HUB_CACHE", str(hub_dir))
    monkeypatch.setenv("HF_HOME", str(home_dir))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved.startswith(str(hub_dir))
    assert str(home_dir) not in resolved


def test_hunyuan_default_cache_dir_honours_legacy_alias(monkeypatch, tmp_path):
    """``HUGGINGFACE_HUB_CACHE`` is huggingface_hub's legacy alias for
    ``HF_HUB_CACHE`` (both still documented + supported upstream — no
    formal deprecation); long-standing cluster module files often set
    only the old name. The helper must respect it when ``HF_HUB_CACHE``
    is unset."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved == str(
        tmp_path / "paintbench" / "HunyuanImage-3-Instruct-Distil"
    )


def test_hunyuan_default_cache_dir_new_name_wins_over_legacy(monkeypatch, tmp_path):
    """When both names are set, the new one wins — matches huggingface_hub's
    own behaviour (the legacy var is a fallback only)."""
    _isolated_hf_env(monkeypatch)
    new_dir = tmp_path / "new"
    old_dir = tmp_path / "old"
    monkeypatch.setenv("HF_HUB_CACHE", str(new_dir))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(old_dir))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved.startswith(str(new_dir))
    assert str(old_dir) not in resolved


def test_hunyuan_default_cache_dir_falls_back_to_hf_home(monkeypatch, tmp_path):
    """No HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE → ``${HF_HOME}/hub``."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved == str(
        tmp_path / "hub" / "paintbench" / "HunyuanImage-3-Instruct-Distil"
    )


def test_hunyuan_default_cache_dir_honours_xdg_cache_home(monkeypatch, tmp_path):
    """``XDG_CACHE_HOME`` is huggingface_hub's own next-to-last fallback
    when no HF cache var is set. Clusters/containers that redirect
    application caches via the XDG base-dir spec rather than
    HF-specific vars still need to land in the right cache, otherwise
    the ~160 GB Hunyuan snapshot lands in $HOME and defeats the
    redirect."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved == str(
        tmp_path
        / "huggingface"
        / "hub"
        / "paintbench"
        / "HunyuanImage-3-Instruct-Distil"
    )


def test_hunyuan_default_cache_dir_hf_home_wins_over_xdg(monkeypatch, tmp_path):
    """When both are set, ``HF_HOME`` takes precedence over
    ``XDG_CACHE_HOME`` — matches huggingface_hub's own resolution order
    (HF-specific vars beat the generic XDG redirect)."""
    _isolated_hf_env(monkeypatch)
    hf_home = tmp_path / "hf_home"
    xdg = tmp_path / "xdg"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(xdg))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved.startswith(str(hf_home))
    assert str(xdg) not in resolved


def test_hunyuan_default_cache_dir_falls_back_to_user_home(monkeypatch, tmp_path):
    """Nothing in env → ``~/.cache/huggingface/hub`` (mirror HF's own
    default). Redirect ``$HOME`` so we don't write into the real one."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = inference.HunyuanImage3InstructDistilModel._default_model_cache_dir()

    assert resolved == str(
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "paintbench"
        / "HunyuanImage-3-Instruct-Distil"
    )


# ── BAGEL upstream-clone resolution ──────────────────────────────────────────
# BAGELModel auto-clones https://github.com/bytedance-seed/BAGEL.git at a
# pinned SHA into a HF-style cache dir on first load_model(). Exercise the
# cache-dir helper (mirror Hunyuan's tests above), the lazy resolution chain
# (kwarg > BAGEL_REPO env > auto-clone fallback), and the _ensure_upstream
# error / short-circuit paths. None of these tests hit the network or call
# git — _ensure_upstream is exercised behind a monkeypatched stub.


def _bare_bagel(*, bagel_repo=None):
    """Build a ``BAGELModel`` instance bypassing __init__.

    BAGELModel.__init__ touches ``torch.cuda.is_available()`` to pick a
    default device, which fails on core-only CI (no torch). The lazy
    upstream-resolution chain in ``_resolve_upstream_dir`` only reads
    ``self.bagel_repo``, so seed it via ``__new__`` and skip the rest.
    Mirrors ``test_hunyuan_forwards_input_size_to_pipeline``'s pattern.
    """
    model = inference.BAGELModel.__new__(inference.BAGELModel)
    model.bagel_repo = bagel_repo
    return model


def test_bagel_registered_and_pinned_to_specific_sha():
    """The pin must be a full SHA (40 hex chars). Catches drive-by edits
    that replace it with a branch name like ``"main"`` — reproducibility
    breaks the moment upstream lands a non-trivial inference-path commit."""
    assert "bagel" in inference._REGISTRY
    assert inference._REGISTRY["bagel"] is inference.BAGELModel
    assert len(inference._BAGEL_UPSTREAM_REF) == 40
    assert all(c in "0123456789abcdef" for c in inference._BAGEL_UPSTREAM_REF)


def test_bagel_default_upstream_cache_dir_uses_hf_hub_cache(monkeypatch, tmp_path):
    """HF_HUB_CACHE wins outright — BAGEL clone lands in
    ``<HF_HUB_CACHE>/paintbench/bagel-upstream``."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))

    resolved = inference.BAGELModel._default_upstream_cache_dir()

    assert resolved == str(tmp_path / "paintbench" / "bagel-upstream")


def test_bagel_default_upstream_cache_dir_falls_back_to_hf_home(monkeypatch, tmp_path):
    """No HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE → ``${HF_HOME}/hub``."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    resolved = inference.BAGELModel._default_upstream_cache_dir()

    assert resolved == str(
        tmp_path / "hub" / "paintbench" / "bagel-upstream"
    )


def test_bagel_default_upstream_cache_dir_falls_back_to_user_home(monkeypatch, tmp_path):
    """Nothing in env → ``~/.cache/huggingface/hub/paintbench/bagel-upstream``."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HOME", str(tmp_path))

    resolved = inference.BAGELModel._default_upstream_cache_dir()

    assert resolved == str(
        tmp_path
        / ".cache"
        / "huggingface"
        / "hub"
        / "paintbench"
        / "bagel-upstream"
    )


def test_bagel_resolve_upstream_prefers_kwarg_over_env_and_auto_clone(monkeypatch, tmp_path):
    """Resolution precedence: __init__ kwarg (= --bagel-repo) >
    BAGEL_REPO env > _ensure_upstream(default cache dir). When the
    kwarg is set, _ensure_upstream must NOT be called."""
    monkeypatch.setenv("BAGEL_REPO", str(tmp_path / "from_env"))

    called = []
    monkeypatch.setattr(
        inference.BAGELModel,
        "_ensure_upstream",
        classmethod(lambda cls, target: called.append(target) or target),
    )

    explicit = str(tmp_path / "from_kwarg")
    model = _bare_bagel(bagel_repo=explicit)
    resolved = model._resolve_upstream_dir()

    assert resolved == explicit
    assert model.bagel_repo == explicit
    assert called == [], "kwarg path must not invoke _ensure_upstream"


def test_bagel_resolve_upstream_falls_back_to_env(monkeypatch, tmp_path):
    """No kwarg but BAGEL_REPO env set → env wins over auto-clone."""
    env_path = str(tmp_path / "from_env")
    monkeypatch.setenv("BAGEL_REPO", env_path)

    called = []
    monkeypatch.setattr(
        inference.BAGELModel,
        "_ensure_upstream",
        classmethod(lambda cls, target: called.append(target) or target),
    )

    model = _bare_bagel()
    resolved = model._resolve_upstream_dir()

    assert resolved == env_path
    assert model.bagel_repo == env_path
    assert called == [], "env path must not invoke _ensure_upstream"


def test_bagel_resolve_upstream_auto_clones_when_no_override(monkeypatch, tmp_path):
    """No kwarg, no env → _ensure_upstream(default cache dir).
    Verify the resolved default lands in the patched HF_HUB_CACHE root."""
    _isolated_hf_env(monkeypatch)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    monkeypatch.delenv("BAGEL_REPO", raising=False)

    captured = {}

    def _stub_ensure(cls, target):
        captured["target"] = target
        return target

    monkeypatch.setattr(
        inference.BAGELModel, "_ensure_upstream", classmethod(_stub_ensure)
    )

    model = _bare_bagel()
    resolved = model._resolve_upstream_dir()

    expected = str(tmp_path / "paintbench" / "bagel-upstream")
    assert captured["target"] == expected
    assert resolved == expected
    assert model.bagel_repo == expected


def test_bagel_ensure_upstream_errors_clearly_without_git(monkeypatch, tmp_path):
    """Missing ``git`` on PATH should produce a clear, user-actionable
    error rather than an opaque ``subprocess.CalledProcessError``."""
    monkeypatch.setattr(inference.shutil, "which", lambda binary: None)

    with pytest.raises(RuntimeError, match="git not found on PATH"):
        inference.BAGELModel._ensure_upstream(str(tmp_path / "bagel"))


def test_bagel_ensure_upstream_short_circuits_when_sha_matches(monkeypatch, tmp_path):
    """When ``target/.git`` already exists and HEAD matches the pin,
    ``_ensure_upstream`` returns immediately without touching the remote
    (single ``git rev-parse HEAD`` call)."""
    target = tmp_path / "bagel"
    (target / ".git").mkdir(parents=True)
    monkeypatch.setattr(inference.shutil, "which", lambda binary: "/usr/bin/git")

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(tuple(cmd))
        if cmd[:3] == ["git", "-C", str(target)] and cmd[3:5] == ["rev-parse", "HEAD"]:
            class _Result:
                stdout = inference._BAGEL_UPSTREAM_REF + "\n"
                stderr = ""
                returncode = 0
            return _Result()
        raise AssertionError(
            f"unexpected git call after SHA-matches short-circuit: {cmd}"
        )

    monkeypatch.setattr(inference.subprocess, "run", _fake_run)

    result = inference.BAGELModel._ensure_upstream(str(target))

    assert result == str(target)
    assert len(calls) == 1
    assert calls[0][:5] == ("git", "-C", str(target), "rev-parse", "HEAD")


def test_bagel_ensure_upstream_fresh_clone_failure_leaves_target_untouched(
    monkeypatch, tmp_path
):
    """If ``git fetch`` fails mid-clone, the production ``target/`` dir
    must NOT end up with a half-initialized ``.git`` that would trip the
    rev-parse HEAD branch on re-entry. The staging-then-os.replace
    approach guarantees this: failure leaves debris only at
    ``target.tmp``, never at ``target``.

    Regression test for the partial-clone footgun in the BAGEL upstream
    auto-clone path.
    """
    target = tmp_path / "bagel"
    monkeypatch.setattr(inference.shutil, "which", lambda binary: "/usr/bin/git")

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append(tuple(cmd))
        # Let init + remote add succeed against the staging dir; fail on fetch.
        if cmd[1] == "init" or cmd[3:5] == ["remote", "add"]:
            # `git init <path>` actually creates the .git dir on disk; the
            # real subprocess.run would do it via the git binary. Mirror that
            # so the staging dir looks "started".
            if cmd[1] == "init":
                Path(cmd[-1], ".git").mkdir(parents=True, exist_ok=True)

            class _Ok:
                stdout = ""
                stderr = ""
                returncode = 0

            return _Ok()
        if cmd[3] == "fetch":
            raise subprocess.CalledProcessError(
                128, cmd, output="", stderr="fatal: simulated network failure"
            )
        raise AssertionError(f"unexpected git call: {cmd}")

    monkeypatch.setattr(inference.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="git .*fetch.*failed"):
        inference.BAGELModel._ensure_upstream(str(target))

    # The crux: production target dir must not have a .git (or even exist).
    # All debris is confined to the .tmp staging dir.
    assert not (target / ".git").exists(), (
        f"fetch failure left a half-initialized .git at {target} — "
        "the staging-then-os.replace contract is broken"
    )
    assert not target.exists() or not any(target.iterdir()), (
        f"fetch failure polluted {target} with non-staging content"
    )
