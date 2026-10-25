"""Sensory ingress for the cascade — upload, classify, encode, route.

This is the missing front door for A2/P6: a caller can hand the cascade one
image, audio clip, video, or text document, and get back a content id, the
detected modality, a stored copy, an embedding, and the path the scheduler
would take.

The module deliberately contains **no model code and no model import**.  The
encoder is an injected :class:`SensoryEncoder`; :class:`MockSensoryEncoder`
is deterministic and cheap enough for unit tests.  A real multimodal encoder
(for example EmbeddingGemma 2) belongs behind the same protocol and must load
lazily, outside this module, once its memory budget is measured.

Audio and video are accepted and stored today but have no cascade path yet;
``bucket_for_modality`` returns ``None`` for them so the API can say so
instead of pretending P6 handles them.
"""

from __future__ import annotations

import hashlib
import math
import mimetypes
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


class Modality(str, Enum):
    """The sensory channels the ingress understands."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    UNKNOWN = "unknown"


#: MIME types we treat as text even though they are under ``application/``.
_TEXT_APPLICATION_TYPES = frozenset(
    {
        "application/json",
        "application/jsonl",
        "application/ld+json",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
        "application/toml",
        "application/javascript",
        "application/x-javascript",
        "application/typescript",
        "application/sql",
        "application/x-sh",
        "application/x-httpd-php",
    }
)

_TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".text", ".md", ".markdown", ".rst", ".json", ".jsonl",
        ".ndjson", ".yaml", ".yml", ".toml", ".py", ".pyi", ".js", ".mjs",
        ".cjs", ".ts", ".tsx", ".jsx", ".c", ".h", ".hpp", ".cpp", ".cc",
        ".rs", ".go", ".java", ".kt", ".swift", ".rb", ".php", ".sh",
        ".bash", ".zsh", ".sql", ".html", ".htm", ".css", ".xml", ".csv",
        ".tsv", ".log",
    }
)
_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
     ".avif", ".heic", ".heif"}
)
_AUDIO_SUFFIXES = frozenset(
    {".mp3", ".wav", ".ogg", ".oga", ".flac", ".m4a", ".aac", ".opus"}
)
_VIDEO_SUFFIXES = frozenset(
    {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg"}
)

#: Buckets that already have a cascade path.  Only vision is wired for a
#: non-text modality today (P6); the rest are ingest-only.
SENSE_BUCKETS: Mapping[Modality, str] = {
    Modality.TEXT: "qa",
    Modality.IMAGE: "vision",
}


class SensesError(ValueError):
    """A sensory payload is malformed or cannot be accepted."""


class UnsupportedModality(SensesError):
    """The payload is not text, image, audio, or video."""


class PayloadTooLarge(SensesError):
    """The upload exceeds the service's byte budget."""


class EncoderError(SensesError):
    """The injected encoder could not produce a usable vector."""


def _clean_mime(mime: str | None) -> str:
    if not mime:
        return ""
    return str(mime).split(";", 1)[0].strip().lower()


def _from_mime(mime: str) -> Modality | None:
    if mime.startswith("image/"):
        return Modality.IMAGE
    if mime.startswith("audio/"):
        return Modality.AUDIO
    if mime.startswith("video/"):
        return Modality.VIDEO
    if mime.startswith("text/") or mime in _TEXT_APPLICATION_TYPES:
        return Modality.TEXT
    return None


def _from_suffix(filename: str | None) -> Modality | None:
    if not filename:
        return None
    suffix = Path(str(filename)).suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return Modality.TEXT
    if suffix in _IMAGE_SUFFIXES:
        return Modality.IMAGE
    if suffix in _AUDIO_SUFFIXES:
        return Modality.AUDIO
    if suffix in _VIDEO_SUFFIXES:
        return Modality.VIDEO
    return None


def sniff_modality(data: bytes) -> Modality:
    """Recognise a few common container signatures without a codec.

    This only saves uploads whose MIME is generic or absent; it is not a
    media parser and never inspects the payload beyond the header.
    """
    head = data[:32]
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return Modality.IMAGE
    if head.startswith(b"\xff\xd8\xff") or head.startswith((b"GIF87a", b"GIF89a")):
        return Modality.IMAGE
    if head.startswith(b"BM") or head.startswith((b"II*\x00", b"MM\x00*")):
        return Modality.IMAGE
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return Modality.IMAGE
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return Modality.AUDIO
    if head.startswith(b"fLaC") or head.startswith(b"OggS"):
        return Modality.AUDIO
    if head.startswith(b"ID3") or (
        len(head) > 1 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0
    ):
        return Modality.AUDIO
    if head[4:8] == b"ftyp" or head.startswith(b"\x1aE\xdf\xa3"):
        return Modality.VIDEO
    return Modality.UNKNOWN


def detect_modality(
    content_type: str | None,
    filename: str | None = None,
    data: bytes | None = None,
) -> Modality:
    """Classify a payload by MIME first, then by filename, then by header.

    A generic ``application/octet-stream`` does not hide a useful suffix: we
    fall through to the name, then to :mod:`mimetypes`, then to a small
    magic-byte check.  Anything still unknown is :attr:`Modality.UNKNOWN` and
    should be rejected by the caller.
    """
    clean = _clean_mime(content_type)
    found = _from_mime(clean)
    if found is not None:
        return found

    found = _from_suffix(filename)
    if found is not None:
        return found

    guessed = _clean_mime(mimetypes.guess_type(filename or "")[0])
    found = _from_mime(guessed)
    if found is not None:
        return found
    return sniff_modality(data or b"")


def guess_mime(content_type: str | None, filename: str | None = None) -> str:
    """Best available MIME value for storage metadata."""
    clean = _clean_mime(content_type)
    if clean and clean != "application/octet-stream":
        return clean
    guessed = _clean_mime(mimetypes.guess_type(filename or "")[0])
    return guessed or clean or "application/octet-stream"


def bucket_for_modality(modality: Modality | str) -> str | None:
    """The C1 bucket a modality maps to, or ``None`` when no path exists yet."""
    value = modality if isinstance(modality, Modality) else Modality(str(modality))
    return SENSE_BUCKETS.get(value)


@dataclass(frozen=True)
class Sense:
    """One accepted sensory input and its embedding."""

    id: str
    modality: Modality
    mime: str
    size_bytes: int
    sha256: str
    embedding: tuple[float, ...]
    model: str
    dimensions: int
    stored_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "modality": self.modality.value,
            "mime": self.mime,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "embedding": list(self.embedding),
            "dimensions": self.dimensions,
            "model": self.model,
            "stored_path": self.stored_path,
            "metadata": dict(self.metadata),
        }


class SensoryEncoder(Protocol):
    """A cheap boundary around any image/audio/video/text encoder.

    ``encode`` must be side-effect free and return a finite vector.  A real
    implementation may load its weights lazily; this module never calls it at
    import time.
    """

    model: str
    dimensions: int

    def encode(
        self,
        data: bytes,
        *,
        modality: Modality,
        mime: str,
        filename: str | None = None,
    ) -> Sequence[float]:
        ...

    def status(self) -> dict[str, Any]:
        ...


class MockSensoryEncoder:
    """Deterministic, model-free encoder for tests and contract wiring."""

    model = "mock-senses-v1"
    dimensions = 8

    def encode(
        self,
        data: bytes,
        *,
        modality: Modality,
        mime: str,
        filename: str | None = None,
    ) -> Sequence[float]:
        payload = b"\0".join(
            (
                modality.value.encode("utf-8"),
                mime.encode("utf-8"),
                (filename or "").encode("utf-8"),
                data,
            )
        )
        digest = hashlib.sha256(payload).digest()
        return tuple(
            round(int.from_bytes(digest[i : i + 2], "big") / 32767.5 - 1.0, 6)
            for i in range(0, self.dimensions * 2, 2)
        )

    def status(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "loaded": False,
            "kind": "mock",
        }


class SensesService:
    """Validate, store, encode and classify one sensory upload.

    The service is deliberately synchronous and model-agnostic.  The API
    dispatches it through a worker thread so a future local encoder cannot
    block the event loop.  Nothing here decides an answer; it only produces
    the inputs the cascade's P6 path needs.
    """

    def __init__(
        self,
        encoder: SensoryEncoder,
        *,
        store_dir: str | Path | None = None,
        max_bytes: int = 32 * 1024 * 1024,
        max_dimensions: int = 4096,
        allowed: Iterable[Modality | str] = (
            Modality.TEXT,
            Modality.IMAGE,
            Modality.AUDIO,
            Modality.VIDEO,
        ),
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be > 0")
        if max_dimensions <= 0:
            raise ValueError("max_dimensions must be > 0")
        if int(encoder.dimensions) <= 0:
            raise ValueError("encoder.dimensions must be > 0")
        if int(encoder.dimensions) > max_dimensions:
            raise ValueError(
                f"encoder dimensions {encoder.dimensions} exceed {max_dimensions}"
            )
        self.encoder = encoder
        self.store_dir = Path(store_dir) if store_dir is not None else None
        self.max_bytes = int(max_bytes)
        self.max_dimensions = int(max_dimensions)
        self.allowed = frozenset(
            item if isinstance(item, Modality) else Modality(str(item))
            for item in allowed
        )

    def status(self) -> dict[str, Any]:
        base = {
            "model": self.encoder.model,
            "dimensions": int(self.encoder.dimensions),
            "max_bytes": self.max_bytes,
            "max_dimensions": self.max_dimensions,
            "modalities": [m.value for m in Modality if m in self.allowed],
            "store_enabled": self.store_dir is not None,
            "store_dir": str(self.store_dir) if self.store_dir is not None else None,
        }
        try:
            extra = self.encoder.status()
        except Exception:  # noqa: BLE001 - status must never take down a health check
            extra = {}
        return {**base, **extra}

    def ingest(
        self,
        *,
        data: bytes,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> Sense:
        if not data:
            raise SensesError("empty upload")
        if len(data) > self.max_bytes:
            raise PayloadTooLarge(
                f"upload is {len(data)} bytes; limit is {self.max_bytes}"
            )

        modality = detect_modality(content_type, filename, data)
        if modality is Modality.UNKNOWN:
            raise UnsupportedModality(
                "unsupported media type; expected text, image, audio, or video"
            )
        if modality not in self.allowed:
            raise UnsupportedModality(f"{modality.value} input is not enabled")

        mime = guess_mime(content_type, filename)
        try:
            raw_embedding = self.encoder.encode(
                data, modality=modality, mime=mime, filename=filename
            )
        except SensesError:
            raise
        except Exception as exc:  # noqa: BLE001 - an encoder failure is a 503 at the edge
            raise EncoderError(f"encoder failed: {exc}") from exc

        embedding = tuple(float(value) for value in raw_embedding)
        if not embedding:
            raise EncoderError("encoder returned an empty vector")
        if any(not math.isfinite(value) for value in embedding):
            raise EncoderError("encoder returned a non-finite vector")
        if len(embedding) != int(self.encoder.dimensions):
            raise EncoderError(
                "encoder dimension mismatch: "
                f"expected {self.encoder.dimensions}, got {len(embedding)}"
            )

        digest = hashlib.sha256(data).hexdigest()
        sense_id = f"{modality.value}-{digest[:16]}"
        stored_path = self._store(data, filename=filename, sense_id=sense_id)
        return Sense(
            id=sense_id,
            modality=modality,
            mime=mime,
            size_bytes=len(data),
            sha256=digest,
            embedding=embedding,
            model=self.encoder.model,
            dimensions=len(embedding),
            stored_path=stored_path,
            metadata={
                "filename": filename or "",
                "bucket": bucket_for_modality(modality) or "",
            },
        )

    def _store(
        self,
        data: bytes,
        *,
        filename: str | None,
        sense_id: str,
    ) -> str | None:
        if self.store_dir is None:
            return None
        suffix = Path(filename).suffix.lower() if filename else ""
        if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
            suffix = ".bin"
        target_dir = self.store_dir / sense_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"source{suffix}"
        target.write_bytes(data)
        return str(target)
