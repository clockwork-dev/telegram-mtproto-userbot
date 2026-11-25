"""Hot SQLite memory database backup, retention pruning, and safe restore."""

from __future__ import annotations

import asyncio
import gzip
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union


@dataclass(frozen=True)
class BackupResult:
    """Metadata describing a completed SQLite memory backup."""

    backup_path: Path
    source_db_path: Path
    size_bytes: int
    compressed: bool
    created_at: datetime
    pruned_count: int = 0


def _chmod_safe(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _verify_sqlite_integrity(db_path: Path) -> None:
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("PRAGMA quick_check;").fetchone()
    if not row or str(row[0]).lower() != "ok":
        raise RuntimeError(f"SQLite integrity check failed for {db_path}: {row}")


def _unique_backup_path(backup_dir: Path, stamp: str, ext: str) -> Path:
    candidate = backup_dir / f"memory-{stamp}{ext}"
    if not candidate.exists():
        return candidate
    for idx in range(1, 1000):
        numbered = backup_dir / f"memory-{stamp}-{idx:03d}{ext}"
        if not numbered.exists():
            return numbered
    raise RuntimeError(f"Could not allocate unique backup filename in {backup_dir}")


def list_backups(backup_dir: Union[str, Path]) -> List[Path]:
    """Return available memory backups sorted newest-first."""
    directory = Path(backup_dir)
    if not directory.exists() or not directory.is_dir():
        return []
    files = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.name.startswith("memory-")
        and (p.name.endswith(".sqlite") or p.name.endswith(".sqlite.gz"))
    ]
    files.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
    return files


def prune_old_backups(
    backup_dir: Union[str, Path],
    retain: int,
    *,
    exclude: Union[str, Path, None] = None,
) -> int:
    """Delete oldest backups beyond *retain* and return number of deleted files."""
    if retain <= 0:
        return 0
    backups = list_backups(backup_dir)
    if len(backups) <= retain:
        return 0
    exclude_resolved = Path(exclude).resolve() if exclude is not None else None
    pruned = 0
    for old_path in backups[retain:]:
        if exclude_resolved is not None and old_path.resolve() == exclude_resolved:
            continue
        try:
            old_path.unlink()
            pruned += 1
        except OSError:
            pass
    return pruned


def backup_sqlite_database(
    db_path: Union[str, Path],
    backup_dir: Union[str, Path],
    *,
    compress: bool = True,
    retain: int = 14,
) -> BackupResult:
    """Create a consistent hot snapshot of a live SQLite database using sqlite3.backup()."""
    src = Path(db_path)
    if not src.exists():
        raise FileNotFoundError(f"Memory database does not exist: {src}")

    dst_dir = Path(backup_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    _chmod_safe(dst_dir, 0o700)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%d-%H%M%S-%f")
    ext = ".sqlite.gz" if compress else ".sqlite"
    final_path = _unique_backup_path(dst_dir, stamp, ext)
    tmp_sqlite = dst_dir / f".tmp-memory-{stamp}-{os.getpid()}.sqlite"

    try:
        with sqlite3.connect(f"file:{src.resolve()}?mode=ro", uri=True, timeout=30.0) as src_conn:
            with sqlite3.connect(str(tmp_sqlite)) as dst_conn:
                src_conn.backup(dst_conn)
        _verify_sqlite_integrity(tmp_sqlite)

        if compress:
            tmp_gz = dst_dir / f".tmp-memory-{stamp}-{os.getpid()}.sqlite.gz"
            try:
                with tmp_sqlite.open("rb") as f_in, gzip.open(tmp_gz, "wb", compresslevel=6) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                _chmod_safe(tmp_gz, 0o600)
                tmp_gz.replace(final_path)
            finally:
                if tmp_gz.exists():
                    tmp_gz.unlink(missing_ok=True)
        else:
            _chmod_safe(tmp_sqlite, 0o600)
            tmp_sqlite.replace(final_path)
    finally:
        if tmp_sqlite.exists():
            tmp_sqlite.unlink(missing_ok=True)

    size_bytes = final_path.stat().st_size
    pruned_count = prune_old_backups(dst_dir, retain=retain, exclude=final_path)
    return BackupResult(
        backup_path=final_path,
        source_db_path=src,
        size_bytes=size_bytes,
        compressed=compress,
        created_at=now,
        pruned_count=pruned_count,
    )


def restore_sqlite_database(
    backup_path: Union[str, Path],
    target_db_path: Union[str, Path],
) -> Path:
    """Safely verify and restore a SQLite memory backup into *target_db_path*."""
    src_backup = Path(backup_path)
    if not src_backup.exists():
        raise FileNotFoundError(f"Backup file does not exist: {src_backup}")

    target = Path(target_db_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_unpacked = target.parent / f".tmp-restore-{os.getpid()}.sqlite"
    try:
        if src_backup.name.endswith(".gz"):
            with gzip.open(src_backup, "rb") as f_in, tmp_unpacked.open("wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src_backup, tmp_unpacked)

        _verify_sqlite_integrity(tmp_unpacked)

        if target.exists():
            pre_restore = target.with_name(f"{target.name}.pre-restore")
            with sqlite3.connect(str(target), timeout=30.0) as cur_conn:
                with sqlite3.connect(str(pre_restore)) as pre_conn:
                    cur_conn.backup(pre_conn)
            _chmod_safe(pre_restore, 0o600)

        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{target}{suffix}")
            if sidecar.exists():
                sidecar.unlink()

        _chmod_safe(tmp_unpacked, 0o600)
        tmp_unpacked.replace(target)
        return target
    finally:
        if tmp_unpacked.exists():
            tmp_unpacked.unlink(missing_ok=True)


async def async_backup_sqlite_database(
    db_path: Union[str, Path],
    backup_dir: Union[str, Path],
    *,
    compress: bool = True,
    retain: int = 14,
) -> BackupResult:
    """Async wrapper around :func:`backup_sqlite_database`."""
    return await asyncio.to_thread(
        backup_sqlite_database,
        db_path,
        backup_dir,
        compress=compress,
        retain=retain,
    )


async def async_restore_sqlite_database(
    backup_path: Union[str, Path],
    target_db_path: Union[str, Path],
) -> Path:
    """Async wrapper around :func:`restore_sqlite_database`."""
    return await asyncio.to_thread(restore_sqlite_database, backup_path, target_db_path)
