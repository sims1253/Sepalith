"""Manifests for learned latent artifacts; no tensor or inference dependency."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

MEMORY_SCHEMA_VERSION = "sepalith.latent-memory.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REPRESENTATIONS = {"input_embeddings", "kv_cache"}
_DTYPES = {"float16", "bfloat16", "float32"}


def _identity(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} requires a nonempty identity")
    return value


def _relative_file(value: object, name: str) -> None:
    value = _identity(value, name)
    path = PurePosixPath(value)
    if (path.is_absolute() or "\\" in value or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError(f"{name} requires a normalized relative POSIX file path")


def _digest(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} requires a lowercase SHA256 hex digest")


@dataclass(frozen=True)
class ConsumerCompatibility:
    """Exact compatibility identities configured by the consumer, not guessed."""

    encoder_revision: str
    decoder_revision: str
    tokenizer_revision: str
    latent_count: int
    latent_width: int
    dtype: str
    layout_contract: str
    scope: str
    representation: str = "input_embeddings"

    def __post_init__(self) -> None:
        for name in ("encoder_revision", "decoder_revision", "tokenizer_revision",
                     "layout_contract", "scope"):
            _identity(getattr(self, name), name)
        for name in ("latent_count", "latent_width"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.representation not in _REPRESENTATIONS:
            raise ValueError("Unsupported latent representation")
        if self.dtype not in _DTYPES:
            raise ValueError("Unsupported latent dtype")


@dataclass(frozen=True)
class LatentMemoryManifest:
    compatibility: ConsumerCompatibility
    payload_path: str
    payload_sha256: str
    source_hashes: Mapping[str, str]
    schema_version: str = MEMORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMORY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported memory schema_version: {self.schema_version!r}")
        if not isinstance(self.compatibility, ConsumerCompatibility):
            raise ValueError("compatibility must be ConsumerCompatibility")
        _relative_file(self.payload_path, "payload_path")
        _digest(self.payload_sha256, "payload_sha256")
        if not isinstance(self.source_hashes, Mapping) or not self.source_hashes:
            raise ValueError("source_hashes must identify at least one workspace file")
        for path, digest in self.source_hashes.items():
            _relative_file(path, "source path")
            _digest(digest, "source hash")
        # Copy and freeze the hashes: later caller mutation must not change provenance.
        from types import MappingProxyType
        object.__setattr__(self, "source_hashes", MappingProxyType(dict(self.source_hashes)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "compatibility": asdict(self.compatibility),
            "payload_path": self.payload_path,
            "payload_sha256": self.payload_sha256,
            "source_hashes": dict(self.source_hashes),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> LatentMemoryManifest:
        required = {"schema_version", "compatibility", "payload_path", "payload_sha256", "source_hashes"}
        if not isinstance(record, Mapping) or set(record) != required:
            raise ValueError("Memory manifest requires exactly the v1 schema fields")
        compatibility = record["compatibility"]
        fields = set(ConsumerCompatibility.__dataclass_fields__)
        if not isinstance(compatibility, Mapping) or set(compatibility) != fields:
            raise ValueError("compatibility requires exactly the v1 compatibility fields")
        return cls(compatibility=ConsumerCompatibility(**compatibility),
                   payload_path=record["payload_path"], payload_sha256=record["payload_sha256"],
                   source_hashes=record["source_hashes"], schema_version=record["schema_version"])

    def stale_sources(self, current_hashes: Mapping[str, str]) -> tuple[str, ...]:
        """Return added, changed or missing files in the complete scoped inventory.

        The caller supplies all current files in the manifest's scope using the
        same inclusion rules as the producer. No filesystem reads occur here.
        """
        if not isinstance(current_hashes, Mapping):
            raise ValueError("current_hashes must be a mapping")
        for path in current_hashes:
            _relative_file(path, "current source path")
        return tuple(sorted(path for path in self.source_hashes.keys() | current_hashes.keys()
                            if path not in self.source_hashes or path not in current_hashes
                            or current_hashes[path] != self.source_hashes[path]))

    def validate_for_consumer(
        self, expected: ConsumerCompatibility, *, current_hashes: Mapping[str, str],
    ) -> None:
        """Fail closed on incompatible identities/shapes or stale source inventory.

        Payload integrity is a separate, explicit verify_payload call. Neither
        method decodes tensors or checks their actual shapes against the manifest.
        """
        if not isinstance(expected, ConsumerCompatibility):
            raise ValueError("expected must be ConsumerCompatibility")
        mismatches = [name for name in ConsumerCompatibility.__dataclass_fields__
                      if getattr(self.compatibility, name) != getattr(expected, name)]
        if mismatches:
            raise ValueError(f"Incompatible latent memory: {', '.join(mismatches)}")
        stale = self.stale_sources(current_hashes)
        if stale:
            raise ValueError(f"Stale latent memory sources: {', '.join(stale)}")

    def verify_payload(self, artifact_directory: str | Path) -> None:
        """Explicitly stream the local payload and compare its SHA256 digest.

        The payload must resolve inside artifact_directory. The consumer must
        separately validate the tensor container and use the verified bytes.
        """
        root = Path(artifact_directory).resolve()
        payload = (root / self.payload_path).resolve()
        if not payload.is_relative_to(root):
            raise ValueError("Payload resolves outside artifact_directory")
        if not payload.is_file():
            raise ValueError("Payload file is missing or is not a regular file")
        digest = hashlib.sha256()
        with payload.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.payload_sha256:
            raise ValueError("Payload SHA256 mismatch")
