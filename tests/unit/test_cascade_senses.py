"""Unit tests for the model-free sensory ingress."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from harness.cascade.senses import (
    EncoderError,
    MockSensoryEncoder,
    Modality,
    PayloadTooLarge,
    SensesError,
    SensesService,
    UnsupportedModality,
    bucket_for_modality,
    detect_modality,
)


@pytest.mark.parametrize(
    ("mime", "filename", "expected"),
    [
        ("image/png", "cat.png", Modality.IMAGE),
        ("image/jpeg; charset=binary", None, Modality.IMAGE),
        ("audio/wav", "clip.wav", Modality.AUDIO),
        ("video/mp4", "clip.mp4", Modality.VIDEO),
        ("text/plain", "notes.txt", Modality.TEXT),
        ("application/json", "data.json", Modality.TEXT),
        ("application/octet-stream", "figure.webp", Modality.IMAGE),
        ("application/octet-stream", "voice.opus", Modality.AUDIO),
        ("application/octet-stream", "notes.md", Modality.TEXT),
        ("application/x-unknown", "blob.bin", Modality.UNKNOWN),
    ],
)
def test_detect_modality(mime, filename, expected):
    assert detect_modality(mime, filename) is expected


def test_magic_bytes_rescue_generic_uploads():
    png = b"\x89PNG\r\n\x1a\n" + b"rest"
    wav = b"RIFF\x00\x00\x00\x00WAVEfmt "
    mp4 = b"\x00\x00\x00\x18ftypisom"
    assert detect_modality("application/octet-stream", "blob", png) is Modality.IMAGE
    assert detect_modality(None, None, wav) is Modality.AUDIO
    assert detect_modality(None, None, mp4) is Modality.VIDEO
    assert detect_modality(None, None, b"\x00\x01\x02") is Modality.UNKNOWN


def test_bucket_mapping_is_explicit_about_unrouted_modalities():
    assert bucket_for_modality(Modality.TEXT) == "qa"
    assert bucket_for_modality(Modality.IMAGE) == "vision"
    assert bucket_for_modality(Modality.AUDIO) is None
    assert bucket_for_modality(Modality.VIDEO) is None


def test_mock_encoder_is_deterministic_and_content_sensitive():
    encoder = MockSensoryEncoder()
    first = encoder.encode(b"abc", modality=Modality.TEXT, mime="text/plain")
    again = encoder.encode(b"abc", modality=Modality.TEXT, mime="text/plain")
    other = encoder.encode(b"abcd", modality=Modality.TEXT, mime="text/plain")
    assert len(first) == 8
    assert tuple(first) == tuple(again)
    assert tuple(first) != tuple(other)
    assert all(-1.0 <= value <= 1.0 for value in first)


def test_ingest_stores_and_classifies_an_image(tmp_path):
    service = SensesService(
        MockSensoryEncoder(), store_dir=tmp_path / "senses", max_bytes=1024
    )
    payload = b"\x89PNG\r\n\x1a\n" + b"not-a-real-image"
    sense = service.ingest(
        data=payload, filename="figure.png", content_type="image/png"
    )
    assert sense.modality is Modality.IMAGE
    assert sense.mime == "image/png"
    assert sense.dimensions == 8
    assert sense.sha256 == hashlib.sha256(payload).hexdigest()
    assert sense.metadata["bucket"] == "vision"
    assert sense.stored_path is not None
    assert Path(sense.stored_path).read_bytes() == payload
    assert Path(sense.stored_path).name == "source.png"


def test_ingest_can_run_without_persisting():
    service = SensesService(MockSensoryEncoder())
    sense = service.ingest(
        data=b"hello", filename="notes.txt", content_type="text/plain"
    )
    assert sense.modality is Modality.TEXT
    assert sense.stored_path is None
    assert sense.metadata["bucket"] == "qa"


def test_ingest_rejects_empty_oversized_and_unknown_payloads():
    service = SensesService(MockSensoryEncoder(), max_bytes=4)
    with pytest.raises(SensesError, match="empty"):
        service.ingest(data=b"", filename="empty.txt", content_type="text/plain")
    with pytest.raises(PayloadTooLarge):
        service.ingest(data=b"12345", filename="x.txt", content_type="text/plain")
    with pytest.raises(UnsupportedModality):
        service.ingest(
            data=b"1234", filename="x.bin", content_type="application/x-unknown"
        )


def test_ingest_rejects_a_modality_not_enabled():
    service = SensesService(MockSensoryEncoder(), allowed=(Modality.TEXT,))
    with pytest.raises(UnsupportedModality, match="image input is not enabled"):
        service.ingest(data=b"image", filename="x.png", content_type="image/png")


def test_status_describes_the_encoder_without_loading_weights(tmp_path):
    service = SensesService(
        MockSensoryEncoder(), store_dir=tmp_path / "senses", max_bytes=16
    )
    status = service.status()
    assert status["model"] == "mock-senses-v1"
    assert status["dimensions"] == 8
    assert status["max_bytes"] == 16
    assert status["max_dimensions"] == 4096
    assert status["store_enabled"] is True
    assert status["loaded"] is False
    assert status["modalities"] == ["text", "image", "audio", "video"]


def test_service_rejects_an_encoder_above_the_dimension_ceiling():
    class WideEncoder:
        model = "wide"
        dimensions = 8

    with pytest.raises(ValueError, match="exceed"):
        SensesService(WideEncoder(), max_dimensions=4)


def test_encoder_failure_is_typed_as_unavailable():
    class BrokenEncoder:
        model = "broken"
        dimensions = 8

        def encode(self, data, *, modality, mime, filename=None):
            raise RuntimeError("out of memory")

        def status(self):
            return {}

    service = SensesService(BrokenEncoder())
    with pytest.raises(EncoderError, match="out of memory"):
        service.ingest(data=b"x", filename="x.txt", content_type="text/plain")
