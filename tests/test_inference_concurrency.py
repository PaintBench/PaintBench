"""Verify the per-instance ``threading.Lock`` in each shared-pipeline
model wrapper actually serialises ``generate()`` under threaded calls.

Regression test for the silent ``IndexError`` the diffusers wrappers
hit when their scheduler's ``_step_index`` races across concurrent
``ThreadPoolExecutor`` workers:

    File "diffusers/schedulers/scheduling_flow_match_euler_discrete.py",
      line 502, in step
        sigma_next = self.sigmas[sigma_idx + 1]
    IndexError: index N+1 is out of bounds for dimension 0 with size N+1

The lock pattern (``with self._lock:`` around the body of
``generate()``) is identical across all 8 shared-pipeline wrappers (6
diffusers, BAGEL, both Hunyuan variants), so a single positive test on
one of them — plus a negative control to prove the probe actually
detects concurrency — is enough to verify the ``BaseModel`` thread-
safety contract.

Uses ``HunyuanImage3InstructDistilModel`` because its ``generate()``
doesn't construct a ``torch.Generator`` (the diffusers wrappers do),
so the test runs on a torch-free install.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import inference


class _ConcurrencyProbe:
    """Fake pipe that records peak in-flight calls. Records the maximum
    number of concurrent ``generate_image()`` calls observed. With the
    per-instance lock in ``BaseModel`` subclasses, this peak is 1."""

    def __init__(self, dwell_s: float = 0.02):
        self.dwell_s = dwell_s
        self._counter_lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0
        self.calls = 0

    def generate_image(self, **_kwargs):
        with self._counter_lock:
            self._in_flight += 1
            if self._in_flight > self.peak_in_flight:
                self.peak_in_flight = self._in_flight
            self.calls += 1
        try:
            # Sleep *outside* the counter lock so the dwell actually
            # overlaps in real time if the model lock is missing —
            # otherwise the counter lock alone would always serialise.
            time.sleep(self.dwell_s)
            return [""], [Image.new("RGB", (4, 4), "white")]
        finally:
            with self._counter_lock:
                self._in_flight -= 1


def _build_hunyuan_with_probe(probe: _ConcurrencyProbe, lock):
    """Bypass ``__init__`` so we don't depend on the upstream snapshot
    download / transformers import. Mirrors the pattern in
    ``test_inference_registry::test_hunyuan_forwards_input_size_to_pipeline``."""
    model = inference.HunyuanImage3InstructDistilModel.__new__(
        inference.HunyuanImage3InstructDistilModel,
    )
    model.pipe = probe
    model.seed = 0
    model.use_system_prompt = False
    model.bot_task = "image"
    model.diff_infer_steps = 8
    model._lock = lock
    return model


def _drive_threaded(model, n_calls: int = 8, workers: int = 4) -> None:
    image = Image.new("RGB", (8, 8), "black")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # list() forces .result() on every future so test-side
        # exceptions surface instead of getting swallowed.
        list(pool.map(lambda _: model.generate(image, "x"), range(n_calls)))


def test_generate_is_serialised_with_lock():
    """Positive: with a real ``threading.Lock``, no two threads ever
    execute the pipeline call concurrently — peak in-flight is 1.
    This is the property that prevents the scheduler-step race."""
    probe = _ConcurrencyProbe(dwell_s=0.02)
    model = _build_hunyuan_with_probe(probe, threading.Lock())

    _drive_threaded(model, n_calls=8, workers=4)

    assert probe.calls == 8
    assert probe.peak_in_flight == 1, (
        f"Lock failed to serialise generate(): observed peak "
        f"{probe.peak_in_flight} concurrent pipeline calls. The shared "
        f"scheduler ``_step_index`` would race here and eventually raise "
        f"``IndexError`` from inside scheduling_flow_match_euler_discrete.step()."
    )


def test_generate_overlaps_without_lock_sanity_check():
    """Negative control: with a no-op context manager swapped in for
    the lock, threads DO overlap inside the pipeline call. Validates
    that the probe actually detects concurrency — without this, a
    silent regression that drops the ``with self._lock:`` line could
    still pass the positive test (e.g. because the GIL happened to
    serialise things on this particular CI host)."""

    class _NoLock:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    probe = _ConcurrencyProbe(dwell_s=0.05)
    model = _build_hunyuan_with_probe(probe, _NoLock())

    _drive_threaded(model, n_calls=8, workers=4)

    assert probe.calls == 8
    assert probe.peak_in_flight >= 2, (
        f"Negative control failed: expected >= 2 concurrent in-flight "
        f"calls with no lock, but observed {probe.peak_in_flight}. The "
        f"probe may not be detecting concurrency correctly, which would "
        f"undermine the positive test."
    )
