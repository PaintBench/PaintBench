"""Config-precedence tests for NanoBanana2Model and the .env helper.

Pins down the CLI > env > class default contract that lets us run different
Gemini variants concurrently from separate terminals (different CLI flags)
while sharing one .env (gateway URL, API key). No API calls — pure unit
tests, run in well under a second."""
from __future__ import annotations

import os

import pytest

import inference


# ─── _load_env_file ──────────────────────────────────────────────────────────

def test_load_env_file_parses_pipes(tmp_path, monkeypatch):
    """`|` in values is preserved (not interpreted as shell pipe). This is
    the bug `set -a; source .env; set +a` hit on values like LLM|abc|def."""
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("LLAMA_API_KEY=LLM|abc|def\n")
    inference._load_env_file(env)
    assert os.environ["LLAMA_API_KEY"] == "LLM|abc|def"


def test_load_env_file_handles_quotes_and_comments(tmp_path, monkeypatch):
    """Surrounding double/single quotes get stripped; comments and blanks ignored."""
    monkeypatch.delenv("X", raising=False)
    monkeypatch.delenv("Y", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "# a comment line\n"
        "\n"
        'X="quoted-value"\n'
        "Y='single-quoted'\n"
    )
    inference._load_env_file(env)
    assert os.environ["X"] == "quoted-value"
    assert os.environ["Y"] == "single-quoted"


def test_load_env_file_preserves_existing_env(tmp_path, monkeypatch):
    """Existing env vars take precedence — shell exports beat .env defaults."""
    monkeypatch.setenv("FOO", "from-shell")
    env = tmp_path / ".env"
    env.write_text("FOO=from-file\n")
    inference._load_env_file(env)
    assert os.environ["FOO"] == "from-shell"


def test_load_env_file_missing_is_no_op(tmp_path):
    """Missing .env is not an error — module-import auto-load can't blow up."""
    inference._load_env_file(tmp_path / "nonexistent.env")  # should not raise


# ─── NanoBanana2Model precedence ─────────────────────────────────────────────

@pytest.fixture
def gemini_clean_env(monkeypatch):
    """Establish a known-clean Gemini env for precedence tests.

    Clears every Gemini-related var the constructor may consult (including
    per-variant ``NB1_*`` / ``NB2_*`` names) so a developer's local ``.env``
    can't influence the test outcome via the module-import auto-loader."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    for var in (
        "GOOGLE_API_KEY",
        "GEMINI_BASE_URL",
        "GEMINI_MODEL_NAME",
        "GEMINI_INCLUDE_THOUGHTS",
        "NB1_MODEL_NAME",
        "NB1_INCLUDE_THOUGHTS",
        "NB2_MODEL_NAME",
        "NB2_INCLUDE_THOUGHTS",
    ):
        monkeypatch.delenv(var, raising=False)


def test_constructor_uses_class_defaults_when_nothing_set(gemini_clean_env):
    """With no env vars and no kwargs, all knobs fall back to class defaults."""
    m = inference.NanoBanana2Model()
    assert m.model_name == "gemini-3.1-flash-image-preview"
    assert m.base_url is None
    assert m.include_thoughts is True


def test_env_vars_override_class_defaults(monkeypatch, gemini_clean_env):
    """Env vars are layer 2: applied when kwargs aren't passed."""
    monkeypatch.setenv("GEMINI_MODEL_NAME", "gemini-foo")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://example.com")
    monkeypatch.setenv("GEMINI_INCLUDE_THOUGHTS", "0")
    m = inference.NanoBanana2Model()
    assert m.model_name == "gemini-foo"
    assert m.base_url == "https://example.com"
    assert m.include_thoughts is False


def test_kwargs_override_env(monkeypatch, gemini_clean_env):
    """Kwargs (the path CLI flags take) win over env. The contract that
    enables concurrent NB1 + NB2 runs sharing one .env."""
    monkeypatch.setenv("GEMINI_MODEL_NAME", "from-env")
    monkeypatch.setenv("GEMINI_BASE_URL", "https://from-env.example")
    monkeypatch.setenv("GEMINI_INCLUDE_THOUGHTS", "0")
    m = inference.NanoBanana2Model(
        model_name="from-cli",
        base_url="https://from-cli.example",
        include_thoughts=True,
    )
    assert m.model_name == "from-cli"
    assert m.base_url == "https://from-cli.example"
    assert m.include_thoughts is True


def test_create_evaluator_forwards_model_name_kwarg(gemini_clean_env):
    """Regression: create_evaluator's first positional arg used to be named
    ``model_name``, which collided with passing a Gemini ``model_name``
    override through **kwargs (the CLI plumbing path). Renamed to
    registry_key; this test pins the contract so it can't drift back."""
    m = inference.create_evaluator(
        "nano-banana-2",
        api_key="test-key",
        model_name="gemini-foo",
        base_url="https://example.com",
        include_thoughts=False,
    )
    assert m.model_name == "gemini-foo"
    assert m.base_url == "https://example.com"
    assert m.include_thoughts is False


# ─── NB1 vs NB2 variant defaults ─────────────────────────────────────────────

def test_nb1_class_defaults_disable_thoughts(gemini_clean_env):
    """NB1 corresponds to Gemini 2.5 Flash Image, which rejects
    thinking_config — class default include_thoughts must be False."""
    m = inference.NanoBanana1Model()
    assert m.model_name == "gemini-2.5-flash-image"
    assert m.include_thoughts is False
    assert m.ENV_KEY == "NB1"


def test_nb2_class_defaults_enable_thoughts(gemini_clean_env):
    """NB2 corresponds to Gemini 3.1 Flash Image Preview, which supports
    thinking_config — default include_thoughts must be True."""
    m = inference.NanoBanana2Model()
    assert m.model_name == "gemini-3.1-flash-image-preview"
    assert m.include_thoughts is True
    assert m.ENV_KEY == "NB2"


# ─── Per-variant env precedence ──────────────────────────────────────────────

def test_per_variant_env_overrides_global_env(monkeypatch, gemini_clean_env):
    """NB1_MODEL_NAME beats GEMINI_MODEL_NAME for NanoBanana1Model — and
    NB2_MODEL_NAME beats GEMINI_MODEL_NAME for NanoBanana2Model. This is
    the contract that lets the two variants share one .env: each picks up
    its own per-variant override, ignoring the legacy global one."""
    monkeypatch.setenv("GEMINI_MODEL_NAME", "global-fallback")
    monkeypatch.setenv("NB1_MODEL_NAME", "nb1-specific")
    monkeypatch.setenv("NB2_MODEL_NAME", "nb2-specific")
    assert inference.NanoBanana1Model().model_name == "nb1-specific"
    assert inference.NanoBanana2Model().model_name == "nb2-specific"


def test_global_env_used_as_fallback_when_per_variant_not_set(
    monkeypatch, gemini_clean_env
):
    """When only the legacy global GEMINI_MODEL_NAME is set, both variants
    fall through to it. Preserves the pre-NB1 behaviour for old .env files."""
    monkeypatch.setenv("GEMINI_MODEL_NAME", "global-only")
    assert inference.NanoBanana1Model().model_name == "global-only"
    assert inference.NanoBanana2Model().model_name == "global-only"


def test_per_variant_include_thoughts_env(monkeypatch, gemini_clean_env):
    """Same precedence logic for include_thoughts: NB2_INCLUDE_THOUGHTS
    overrides class default; GEMINI_INCLUDE_THOUGHTS only used as fallback."""
    monkeypatch.setenv("NB2_INCLUDE_THOUGHTS", "0")
    monkeypatch.setenv("GEMINI_INCLUDE_THOUGHTS", "1")
    # NB2's per-variant env wins over the global
    assert inference.NanoBanana2Model().include_thoughts is False
    # NB1 has no per-variant env, falls through to global env
    assert inference.NanoBanana1Model().include_thoughts is True


def test_kwargs_still_win_over_per_variant_env(monkeypatch, gemini_clean_env):
    """Top of the precedence chain: explicit kwargs (CLI flag path) beat
    even per-variant env vars."""
    monkeypatch.setenv("NB2_MODEL_NAME", "from-env")
    m = inference.NanoBanana2Model(model_name="from-cli")
    assert m.model_name == "from-cli"


def test_concurrent_variants_share_base_url(monkeypatch, gemini_clean_env):
    """base_url is intentionally global only (no per-variant override) —
    both variants typically route through the same gateway."""
    monkeypatch.setenv("GEMINI_BASE_URL", "https://shared-gateway.example/google")
    assert inference.NanoBanana1Model().base_url == "https://shared-gateway.example/google"
    assert inference.NanoBanana2Model().base_url == "https://shared-gateway.example/google"


# ─── --retry-failed helper ───────────────────────────────────────────────────

def test_failed_problem_keys_extracts_only_unsuccessful():
    """Helper for --retry-failed: returns (task_id, index) tuples ONLY for
    problems whose prior run was unsuccessful. Tasks with no failures are
    naturally absent from the returned set."""
    metrics = {
        "tasks": {
            "blending": {"problems": [
                {"index": 0, "success": True,  "inference_time_s": 5.0},
                {"index": 1, "success": False, "error": "transient"},
                {"index": 2, "success": True},
            ]},
            "border": {"problems": [
                {"index": 0, "success": True},
            ]},
            "recolor": {"problems": [
                {"index": 5, "success": False, "error": "x"},
                {"index": 7, "success": False, "error": "y"},
            ]},
        },
    }
    failed = inference._failed_problem_keys(metrics)
    assert failed == {("blending", 1), ("recolor", 5), ("recolor", 7)}


def test_failed_problem_keys_handles_missing_or_empty_metrics():
    """Tolerate metrics JSON without a tasks key, or with empty tasks."""
    assert inference._failed_problem_keys({}) == set()
    assert inference._failed_problem_keys({"tasks": {}}) == set()
    assert inference._failed_problem_keys({"tasks": {"blending": {"problems": []}}}) == set()


@pytest.mark.parametrize("falsy", ["0", "false", "False", "no", ""])
def test_include_thoughts_falsy_spellings_disable(monkeypatch, gemini_clean_env, falsy):
    """Several falsy spellings should disable include_thoughts."""
    monkeypatch.setenv("GEMINI_INCLUDE_THOUGHTS", falsy)
    m = inference.NanoBanana2Model()
    assert m.include_thoughts is False, f"expected False for {falsy!r}"


# ─── Transient-error predicate ───────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    "HTTP 429 too many requests",
    "502 bad gateway",
    "503 service unavailable",
    "504 gateway timeout",
    "httpx.RemoteProtocolError: Server disconnected without sending a response.",
    "ConnectionError: peer reset",
    "ReadTimeout: blah",
    "ConnectTimeout: blah",
    "RemoteDisconnected('foo')",
])
def test_is_transient_error_recognises_known_markers(msg):
    assert inference._is_transient_error(Exception(msg)), \
        f"Should be transient: {msg!r}"


@pytest.mark.parametrize("msg", [
    "ValueError: model_name not in catalog",
    "INVALID_ARGUMENT",
    "401 unauthorized",
    "403 forbidden",
    "AccessDeniedException: model not in entitlement",
])
def test_is_transient_error_rejects_non_transient(msg):
    """Auth/permission/argument errors should not retry — they won't fix
    themselves and retrying just wastes time and quota."""
    assert not inference._is_transient_error(Exception(msg)), \
        f"Should NOT be transient: {msg!r}"
