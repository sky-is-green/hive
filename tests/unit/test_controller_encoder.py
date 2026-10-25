"""The encoder boundary is lazy, finite and honest about availability."""

from __future__ import annotations

import math

import pytest

from harness.controller.encoder import (
    EncoderError,
    EncoderLifecycle,
    EncoderUnavailable,
    SentenceTransformerEncoder,
    encoder_from_settings,
)


class _Clock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class _Timer:
    def __init__(self, interval, function, args=()):
        self.interval = interval
        self.function = function
        self.args = args
        self.started = False
        self.cancelled = False
        self.daemon = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True

    def fire(self):
        if self.started and not self.cancelled:
            self.function(*self.args)


class _FakeModel:
    def __init__(self, rows):
        self.rows = rows

    def get_sentence_embedding_dimension(self):
        return len(self.rows[0]) if self.rows else None

    def encode(self, texts, **kwargs):
        assert kwargs["convert_to_numpy"] is True
        return [list(row) for row in self.rows[:len(texts)]]


def test_status_does_not_load_the_model():
    encoder = SentenceTransformerEncoder("sentence-transformers/paraphrase-MiniLM-L3-v2")
    status = encoder.status()
    assert status["loaded"] is False
    assert status["model"] == "sentence-transformers/paraphrase-MiniLM-L3-v2"
    assert status["local_files_only"] is True


def test_encode_uses_the_fake_backend_and_normalizes_shape():
    encoder = SentenceTransformerEncoder("fake")
    encoder.dimensions = 3
    encoder._model = _FakeModel([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    rows = encoder.encode(["a", "b"])
    assert rows == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    assert encoder.encode_one("a") == [0.1, 0.2, 0.3]


def test_empty_input_is_empty_output():
    encoder = SentenceTransformerEncoder("fake")
    assert encoder.encode([]) == []


def test_non_finite_vector_is_an_error():
    encoder = SentenceTransformerEncoder("fake")
    encoder.dimensions = 2
    encoder._model = _FakeModel([[math.inf, 1.0]])
    with pytest.raises(EncoderError, match="non-finite"):
        encoder.encode(["x"])


def test_dimension_mismatch_is_an_error():
    encoder = SentenceTransformerEncoder("fake")
    encoder.dimensions = 3
    encoder._model = _FakeModel([[1.0, 2.0]])
    with pytest.raises(EncoderError, match="dimension mismatch"):
        encoder.encode(["x"])


def test_empty_model_id_is_rejected():
    with pytest.raises(ValueError, match="model id"):
        SentenceTransformerEncoder("")


def test_served_setting_is_not_silently_treated_as_local():
    with pytest.raises(EncoderUnavailable, match="does not serve"):
        encoder_from_settings({"encoder": {
            "embedding_model": "m", "embedding_backend": "served"}})


def test_local_setting_builds_the_lazy_adapter():
    encoder = encoder_from_settings({"encoder": {
        "embedding_model": "m", "embedding_backend": "local"}})
    assert encoder.model == "m"
    assert encoder.status()["loaded"] is False


def _fake_lifecycle(*, idle_unload=True, idle_seconds=30.0):
    clock = _Clock()
    timers = []
    loads = {"n": 0}

    def factory():
        loads["n"] += 1
        return SentenceTransformerEncoder("fake")

    def timer_factory(interval, function, args=()):
        timer = _Timer(interval, function, args)
        timers.append(timer)
        return timer

    lifecycle = EncoderLifecycle(
        factory, idle_unload=idle_unload, idle_seconds=idle_seconds,
        clock=clock, timer_factory=timer_factory)
    return lifecycle, clock, timers, loads


def test_lifecycle_unloads_after_the_idle_window():
    lifecycle, clock, timers, loads = _fake_lifecycle()
    assert lifecycle.get().model == "fake"
    assert lifecycle.status()["loaded"] is True
    assert len(timers) == 1 and timers[0].started
    assert timers[0].interval == 30.0

    clock.now += 30.0
    timers[0].fire()
    assert lifecycle.status()["loaded"] is False
    assert loads["n"] == 1
    assert lifecycle.get().model == "fake"
    assert loads["n"] == 2


def test_lifecycle_can_disable_idle_unload():
    lifecycle, _clock, timers, _loads = _fake_lifecycle(idle_unload=False)
    lifecycle.get()
    assert timers == []
    assert lifecycle.status()["idle_unload"] is False


def test_touch_rearms_the_timer_and_stale_timer_is_ignored():
    lifecycle, clock, timers, _loads = _fake_lifecycle()
    lifecycle.get()
    first = timers[0]
    clock.now += 10
    lifecycle.touch()
    second = timers[1]
    assert first.cancelled
    clock.now += 30
    first.function(*first.args)  # stale generation must not unload the new window
    assert lifecycle.status()["loaded"] is True
    second.fire()
    assert lifecycle.status()["loaded"] is False


def test_reconfiguring_off_cancels_the_pending_unload():
    lifecycle, _clock, timers, _loads = _fake_lifecycle()
    lifecycle.get()
    lifecycle.configure(idle_unload=False, idle_seconds=30)
    assert timers[0].cancelled
    assert lifecycle.status()["loaded"] is True
    assert lifecycle.status()["idle_remaining_s"] is None
