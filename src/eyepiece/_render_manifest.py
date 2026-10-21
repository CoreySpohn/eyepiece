"""Portable output manifests with deterministic atomic saves."""

import hashlib
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np

import eyepiece
from eyepiece._render_keys import _ffmpeg_version, _hwostyle_version

_MANIFEST_SCHEMA = 1


def _jsonable(value):
    """Copy JSON values, converting NumPy arrays and scalars recursively."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        item = value.item()
        if not isinstance(item, np.generic):
            return _jsonable(item)
    elif value is None or isinstance(value, (str, bool, int)):
        return value
    elif isinstance(value, float) and math.isfinite(value):
        return value
    elif isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {key: _jsonable(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"Unsupported JSON type: {type(value).__name__}")


def _manifest_versions():
    """Collect portable version strings using the content key's ffmpeg probe."""
    return {
        "eyepiece": getattr(eyepiece, "__version__", None),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "hwostyle": _hwostyle_version(),
        "ffmpeg": _ffmpeg_version(),
    }


class Manifest:
    """Record portable output identities and save deterministic JSON atomically.

    A manifest file has a single writer.

    Args:
        path: JSON manifest path.
        root: Folder containing outputs. Defaults to the manifest's parent.
    """

    def __init__(self, path, *, root=None):
        """Create an empty manifest with output paths relative to root."""
        self.path = Path(path)
        self.root = Path(root if root is not None else self.path.parent).resolve()
        self._document = {
            "schema": _MANIFEST_SCHEMA,
            "versions": dict.fromkeys(
                ("eyepiece", "matplotlib", "numpy", "hwostyle", "ffmpeg")
            ),
            "clips": {},
        }
        self._recorded = False
        self._versions_collected = False

    @classmethod
    def load(cls, path, *, root=None) -> "Manifest":
        """Load a manifest, returning an empty one when the file is missing.

        Args:
            path: JSON manifest path.
            root: Folder containing outputs, or None for the manifest's parent.

        Returns:
            A manifest preserving its saved version metadata.

        Raises:
            ValueError: If the file is unreadable, invalid, or unsupported.
        """
        manifest = cls(path, root=root)
        try:
            text = manifest.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return manifest
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"Cannot read manifest {path}: {exc}") from exc
        try:
            document = json.loads(text)
            manifest._validate(document)
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError(f"Invalid manifest {path}: {exc}") from exc
        manifest._document = document
        manifest._versions_collected = True
        return manifest

    def _output_path(self, path):
        """Resolve an output while enforcing containment under the root."""
        path = Path(path)
        absolute = (path if path.is_absolute() else self.root / path).resolve()
        if not absolute.is_relative_to(self.root):
            raise ValueError(f"Output path is outside manifest root: {path}")
        return absolute

    def _validate(self, document):
        """Reject malformed schemas and output paths that escape the root."""
        if not isinstance(document, dict) or (
            type(document.get("schema")) is not int
            or document["schema"] != _MANIFEST_SCHEMA
        ):
            raise ValueError("unsupported schema version")
        versions, clips = document["versions"], document["clips"]
        if not isinstance(versions, dict) or not isinstance(clips, dict):
            raise ValueError("versions and clips must be objects")
        if set(versions) != set(self._document["versions"]) or any(
            value is not None and not isinstance(value, str)
            for value in versions.values()
        ):
            raise ValueError("invalid version metadata")
        for variants in clips.values():
            if not isinstance(variants, dict):
                raise ValueError("clip variants must be objects")
            for entry in variants.values():
                if not isinstance(entry, dict) or not isinstance(entry["key"], str):
                    raise ValueError("entries must have string keys")
                if not isinstance(entry["outputs"], list):
                    raise ValueError("outputs must be a list")
                if entry["frames"] is not None and type(entry["frames"]) is not int:
                    raise ValueError("frames must be an integer or None")
                for field in ("fps", "dpi"):
                    value = entry[field]
                    if value is not None and (
                        type(value) not in (int, float)
                        or (isinstance(value, float) and not math.isfinite(value))
                    ):
                        raise ValueError(f"{field} must be a finite number or None")
                for output in entry["outputs"]:
                    if not isinstance(output, dict):
                        raise ValueError("outputs must be objects")
                    path = output["path"]
                    if not isinstance(path, str) or Path(path).is_absolute():
                        raise ValueError("output paths must be relative strings")
                    self._output_path(path)
                    if (
                        not isinstance(output["sha256"], str)
                        or re.fullmatch(r"[0-9a-f]{64}", output["sha256"]) is None
                        or type(output["bytes"]) is not int
                        or output["bytes"] < 0
                    ):
                        raise ValueError("invalid output hash or size")
                if "extra" in entry and not isinstance(entry["extra"], dict):
                    raise ValueError("extra must be an object")
        _jsonable(document)

    def entry(self, slug, variant) -> dict | None:
        """Return an independent copy of an entry, or None when absent."""
        return deepcopy(self._document["clips"].get(slug, {}).get(variant))

    def discard(self, slug, variant) -> bool:
        """Remove an entry, returning whether it existed.

        Other variants are preserved. Empty clip groups are removed, and the
        next save persists the deletion.

        Args:
            slug: Clip identifier.
            variant: Variant identifier.

        Returns:
            True if an entry was removed, False if it was absent.
        """
        clips = self._document["clips"]
        entries = clips.get(slug)
        if entries is None or variant not in entries:
            return False
        del entries[variant]
        if not entries:
            del clips[slug]
        return True

    def record(
        self,
        slug,
        variant,
        *,
        key,
        outputs,
        frames=None,
        fps=None,
        dpi=None,
        extra=None,
    ):
        """Record output hashes and sizes, replacing an entry after validation.

        Args:
            slug: Clip identifier.
            variant: Variant identifier.
            key: Content key for this variant.
            outputs: Paths, absolute or relative to root, read from disk.
            frames: Integer encoded frame count, or None.
            fps: Finite numeric frame rate, or None.
            dpi: Finite numeric raster resolution, or None.
            extra: Optional JSON dictionary; NumPy values are converted.

        Raises:
            FileNotFoundError: If an output does not exist.
            ValueError: If an output resolves outside root.
            TypeError: If identifiers or metadata cannot be stored as JSON.
        """
        if not all(isinstance(value, str) for value in (slug, variant, key)):
            raise TypeError("slug, variant, and key must be strings")
        if extra is not None and not isinstance(extra, dict):
            raise TypeError("extra must be a JSON dictionary")
        entry = _jsonable({"key": key, "frames": frames, "fps": fps, "dpi": dpi})
        if extra is not None:
            entry["extra"] = _jsonable(extra)
        recorded = []
        for output in outputs:
            path = self._output_path(output)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                size = os.fstat(stream.fileno()).st_size
            recorded.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "sha256": digest,
                    "bytes": size,
                }
            )
        entry["outputs"] = recorded
        self._document["clips"].setdefault(slug, {})[variant] = entry
        self._recorded = True

    def current(self, slug, variant, key) -> bool:
        """Check the key and output sizes without recomputing output hashes."""
        entry = self.entry(slug, variant)
        if entry is None or entry["key"] != key:
            return False
        for output in entry["outputs"]:
            try:
                path = self._output_path(output["path"])
                if not path.is_file() or path.stat().st_size != output["bytes"]:
                    return False
            except (OSError, ValueError):
                return False
        return True

    def verify(self, slug, variant) -> list[str]:
        """List relative paths with missing, unreadable, or changed contents."""
        entry = self.entry(slug, variant)
        failures = []
        for output in entry["outputs"] if entry is not None else ():
            try:
                with self._output_path(output["path"]).open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if digest == output["sha256"]:
                    continue
            except (OSError, ValueError):
                pass
            failures.append(output["path"])
        return failures

    def save(self) -> bool:
        """Atomically save sorted JSON, preserving the file when bytes match.

        Version metadata is collected on the first save and refreshed after
        recording an entry. Loaded versions are preserved otherwise. Parent
        directories are created as needed, and failed writes leave the previous
        file in place. Each save reclaims its deterministic temporary file.

        Returns:
            True if the file was replaced, False if its bytes were unchanged.
        """
        document = deepcopy(self._document)
        if self._recorded or not self._versions_collected:
            document["versions"] = _manifest_versions()
        payload = (json.dumps(document, sort_keys=True, indent=2) + "\n").encode(
            "utf-8"
        )
        try:
            unchanged = self.path.read_bytes() == payload
        except FileNotFoundError:
            unchanged = False
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            if not unchanged:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with temporary.open("wb") as stream:
                    stream.write(payload)
                os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        self._document = document
        self._recorded = False
        self._versions_collected = True
        return not unchanged


# Keep public pickle references stable through the facade.
Manifest.__module__ = "eyepiece.render"
