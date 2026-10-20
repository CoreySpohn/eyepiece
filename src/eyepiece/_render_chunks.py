"""Chunk validation, binary protocol, and workspace ownership."""

import hashlib
import json
import os
import re
import struct
import time
import uuid
import zlib
from contextlib import contextmanager

import numpy as np

_SCHEMA = 1
_SCALE_SCHEMA = 1
_MAGIC = b"EPCHUNK\0"
_HEADER = struct.Struct("!8sI32sIII")
_LENGTH = struct.Struct("!Q")
_MARKER = ".eyepiece-render"
_MARKER_TEXT = "eyepiece render chunks\n"
_CHUNK_NAME = re.compile(r"chunk-[0-9]+\.(?:bin|part|warn(?:\.part)?)\Z")
_LOCK_PART_NAME = re.compile(r"lock-([0-9]+)-[0-9a-f]+\.part\Z")
_LOCK_GRACE_SECONDS = 5
_PROTOCOL_NAME = re.compile(
    r"(?:claim-[0-9a-f]+-[0-9]+|worker-[0-9]+\.(?:ready|part|error(?:\.part)?|log)|"
    r"chunk-[0-9]+\.error(?:\.part)?|job\.pkl|tasks\.(?:pkl|part))\Z"
)


def _identity(key, count, hold, dpi, size, fps, start, stop):
    values = [key, _SCHEMA, count, list(hold), dpi, size, fps, start, stop]
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).digest()


def _delete_chunk(path):
    """Delete a reserved raster chunk together with its scale-check sidecar."""
    path.unlink(missing_ok=True)
    path.with_suffix(".warn").unlink(missing_ok=True)


def _scale_diagnostic(identity, crc, message):
    """Bind a checked chunk's diagnostic to its identity and raster payload."""
    diagnostic = dict(
        schema=_SCALE_SCHEMA, identity=identity.hex(), crc=crc, message=message
    )
    payload = json.dumps(diagnostic, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    diagnostic["checksum"] = hashlib.sha256(payload).hexdigest()
    return diagnostic


def _read_scale_diagnostic(job):
    """Return the checked message, or None for an invalid diagnostic pair."""
    path, identity, start, stop = job
    try:
        with path.open("rb") as stream:
            magic, schema, stored, count, crc, _ = _HEADER.unpack(
                stream.read(_HEADER.size)
            )
        if (magic, schema, stored, count) != (_MAGIC, _SCHEMA, identity, stop - start):
            return None
        diagnostic = json.loads(path.with_suffix(".warn").read_text(encoding="utf-8"))
        if not isinstance(diagnostic, dict) or not isinstance(
            diagnostic.get("message"), str
        ):
            return None
        message = diagnostic["message"]
        return (
            message if diagnostic == _scale_diagnostic(identity, crc, message) else None
        )
    except (FileNotFoundError, UnicodeError, json.JSONDecodeError, struct.error):
        return None


def _chunk_lengths(path, identity, count):
    """Return lengths only if the identity, size, and payload CRC agree."""
    try:
        with path.open("rb") as stream:
            magic, schema, stored, nframes, crc, lengths_crc = _HEADER.unpack(
                stream.read(_HEADER.size)
            )
            if (magic, schema, stored, nframes) != (_MAGIC, _SCHEMA, identity, count):
                return None
            length_data = stream.read(_LENGTH.size * count)
            if len(length_data) != _LENGTH.size * count:
                return None
            if zlib.crc32(length_data) != lengths_crc:
                return None
            lengths = [entry[0] for entry in _LENGTH.iter_unpack(length_data)]
            if any(length == 0 for length in lengths):
                return None
            if path.stat().st_size != stream.tell() + sum(lengths):
                return None
            actual = 0
            while payload := stream.read(1024 * 1024):
                actual = zlib.crc32(payload, actual)
            return lengths if actual == crc else None
    except (FileNotFoundError, struct.error):
        return None


def _read_chunk(job, size):
    path, identity, start, stop = job
    lengths = _chunk_lengths(path, identity, stop - start)
    if lengths is None:
        raise RuntimeError(f"Invalid chunk for frames [{start}, {stop})")
    width, height = size
    with path.open("rb") as stream:
        stream.seek(_HEADER.size + _LENGTH.size * len(lengths))
        for length in lengths:
            raw = zlib.decompress(stream.read(length))
            if len(raw) != width * height * 4:
                raise RuntimeError(f"Invalid raster for frames [{start}, {stop})")
            yield np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


def _pid_alive(pid):
    if pid < 1:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


@contextmanager
def _workspace(folder):
    """Claim an owned directory and release its exclusive pid lock on exit."""
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / _MARKER
    if not marker.exists():
        if any(folder.iterdir()):
            raise RuntimeError(
                f"Refusing non-empty folder without ownership marker: {folder}"
            )
        try:
            with marker.open("x") as stream:
                stream.write(_MARKER_TEXT)
        except FileExistsError:
            pass
    if marker.read_text() != _MARKER_TEXT:
        raise RuntimeError(f"Unrecognized ownership marker in {folder}")
    lock = folder / "render.lock"
    candidate = folder / f"lock-{os.getpid()}-{uuid.uuid4().hex}.part"
    try:
        with candidate.open("x") as stream:
            stream.write(str(os.getpid()))
        while True:
            try:
                # Exclusive linking publishes a fully written pid atomically.
                os.link(candidate, lock)
                break
            except FileExistsError:
                # Serialize stale-lock takeover, checking the inode before
                # deleting so a competing owner cannot lose its lock.
                import fcntl

                try:
                    with lock.open() as stale:
                        fcntl.flock(stale, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        info = os.fstat(stale.fileno())
                        if info.st_ino != lock.stat().st_ino:
                            continue
                        try:
                            pid = int(stale.read())
                        except ValueError:
                            # Older interrupted runs may leave empty locks.
                            # Give a recent writer time to finish first.
                            if time.time() - info.st_mtime < _LOCK_GRACE_SECONDS:
                                raise RuntimeError(
                                    f"Render folder has an unreadable lock: {folder}"
                                ) from None
                        else:
                            if _pid_alive(pid):
                                raise RuntimeError(
                                    f"Render folder is locked by pid {pid}: {folder}"
                                ) from None
                        lock.unlink()
                except FileNotFoundError:
                    continue
                except BlockingIOError:
                    raise RuntimeError(
                        f"Render folder lock is being taken over: {folder}"
                    ) from None
    finally:
        candidate.unlink(missing_ok=True)
    try:
        for path in folder.iterdir():
            lock_part = _LOCK_PART_NAME.fullmatch(path.name)
            if (
                _PROTOCOL_NAME.fullmatch(path.name)
                or (
                    _CHUNK_NAME.fullmatch(path.name)
                    and (
                        path.suffix == ".part"
                        or (
                            path.suffix == ".warn"
                            and not path.with_suffix(".bin").exists()
                        )
                    )
                )
                or (lock_part is not None and not _pid_alive(int(lock_part[1])))
            ):
                path.unlink()
        yield
    finally:
        lock.unlink(missing_ok=True)
