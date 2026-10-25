"""The encoder boundary is lazy, finite and honest about availability."""

from __future__ import annotations

import math

import pytest

from harness.controller.encoder import (
    EncoderError,
    EncoderUnavailable,
    SentenceTransformerEncoder,
    encoder_from_settings,
)


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
