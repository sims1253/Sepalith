from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sepalith.memory import ConsumerCompatibility, LatentMemoryManifest


def sha(text):
    return hashlib.sha256(text).hexdigest()


class MemoryTests(unittest.TestCase):
    def compatibility(self):
        return ConsumerCompatibility(
            encoder_revision="encoder:commit1", decoder_revision="decoder:commit2",
            tokenizer_revision="tokenizer:commit3", latent_count=4, latent_width=16,
            dtype="bfloat16", layout_contract="position-layout:experiment1",
            scope="selected-files:R/a.R,R/b.R",
        )

    def manifest(self):
        return LatentMemoryManifest(self.compatibility(), "latent.bin", sha(b"fake payload"),
                                    {"R/a.R": sha(b"a"), "R/b.R": sha(b"b")})

    def test_compatible_manifest_and_stable_round_trip(self):
        manifest = self.manifest()
        manifest.validate_for_consumer(self.compatibility(), current_hashes=dict(manifest.source_hashes))
        self.assertEqual(manifest.stale_sources(manifest.source_hashes), ())
        self.assertEqual(LatentMemoryManifest.from_dict(json.loads(manifest.to_json())).to_json(),
                         manifest.to_json())
        with self.assertRaises(TypeError):
            manifest.source_hashes["R/a.R"] = sha(b"changed")

    def test_every_compatibility_field_is_checked(self):
        manifest = self.manifest()
        changes = dict(encoder_revision="other", decoder_revision="other", tokenizer_revision="other",
                       latent_count=5, latent_width=32, dtype="float32", layout_contract="other",
                       scope="other", representation="kv_cache")
        for name, value in changes.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                manifest.validate_for_consumer(replace(self.compatibility(), **{name: value}),
                                               current_hashes=manifest.source_hashes)

    def test_source_change_missing_and_malformed_fail_closed(self):
        manifest = self.manifest()
        for current in ({}, {"R/a.R": sha(b"a"), "R/b.R": sha(b"changed")},
                        {"R/a.R": sha(b"a"), "R/b.R": "invalid"}):
            with self.subTest(current=current), self.assertRaisesRegex(ValueError, "Stale"):
                manifest.validate_for_consumer(self.compatibility(), current_hashes=current)
        self.assertEqual(manifest.stale_sources({}), ("R/a.R", "R/b.R"))

    def test_added_source_invalidates_complete_scope_inventory(self):
        manifest = self.manifest()
        current = {**manifest.source_hashes, "new.R": sha(b"new")}
        self.assertEqual(manifest.stale_sources(current), ("new.R",))
        with self.assertRaisesRegex(ValueError, "new.R"):
            manifest.validate_for_consumer(self.compatibility(), current_hashes=current)

    def test_payload_tampering_and_missing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.manifest()
            with self.assertRaisesRegex(ValueError, "missing"):
                manifest.verify_payload(directory)
            path = Path(directory) / "latent.bin"
            path.write_bytes(b"fake payload")
            manifest.verify_payload(directory)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                manifest.verify_payload(directory)

    def test_payload_symlink_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "artifacts"
            root.mkdir()
            outside = Path(directory) / "outside.bin"
            outside.write_bytes(b"fake payload")
            (root / "latent.bin").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "outside"):
                self.manifest().verify_payload(root)

    def test_invalid_shapes_identities_and_paths(self):
        for changes in ({"latent_count": 0}, {"latent_width": True}, {"encoder_revision": ""},
                        {"dtype": "made-up"}, {"representation": "arbitrary_vectors"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.compatibility(), **changes)
        for path in ("../latent.bin", "/tmp/latent.bin", "https://host/payload", "R\\a.R", "./latent.bin"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(self.manifest(), payload_path=path)
        for sources in ({}, {"../a.R": sha(b"a")}, {"a.R": "not-a-hash"}):
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                replace(self.manifest(), source_hashes=sources)

    def test_unknown_fields_and_versions_rejected(self):
        record = self.manifest().to_dict()
        for change in ({"new_field": "unknown"}, {"schema_version": "sepalith.latent-memory.v99"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LatentMemoryManifest.from_dict({**record, **change})
        record["compatibility"]["new_field"] = "unknown"
        with self.assertRaises(ValueError):
            LatentMemoryManifest.from_dict(record)


if __name__ == "__main__":
    unittest.main()
