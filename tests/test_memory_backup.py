"""Unit and CLI tests for hot SQLite memory backup and restore."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path

import pytest

from bot.cli import main as cli_main
from bot.memory.backup import (
    async_backup_sqlite_database,
    backup_sqlite_database,
    list_backups,
    restore_sqlite_database,
)
from bot.memory.db import ChatMessage, MemoryStore


@pytest.mark.asyncio
async def test_hot_backup_wal_and_restore(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    backup_dir = tmp_path / "backups"

    store = MemoryStore(db_path=str(db_path))
    await store.init_db()
    await store.save_message(
        ChatMessage(
            chat_id=-1001,
            telegram_msg_id=1,
            user_id=42,
            author_name="Alice",
            text="Первое сообщение для бэкапа",
            timestamp=1700000001,
        )
    )
    # Keep store open in WAL mode while performing hot backup
    res = await async_backup_sqlite_database(db_path, backup_dir, compress=True, retain=5)
    assert res.backup_path.exists()
    assert res.backup_path.name.endswith(".sqlite.gz")
    assert res.size_bytes > 0
    assert res.compressed is True

    # Add a second message AFTER backup
    await store.save_message(
        ChatMessage(
            chat_id=-1001,
            telegram_msg_id=2,
            user_id=42,
            author_name="Alice",
            text="Сообщение после бэкапа",
            timestamp=1700000002,
        )
    )
    await store.close()

    # Restore from backup -> should have telegram_msg_id=1, not 2, and create .pre-restore
    restored = restore_sqlite_database(res.backup_path, db_path)
    assert restored == db_path
    assert db_path.with_name("memory.sqlite.pre-restore").exists()

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT telegram_msg_id, text FROM messages ORDER BY telegram_msg_id").fetchall()
    assert rows == [(1, "Первое сообщение для бэкапа")]


def test_backup_retention_pruning(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    backup_dir = tmp_path / "backups"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t (val) VALUES ('ok')")

    for _ in range(4):
        backup_sqlite_database(db_path, backup_dir, compress=False, retain=2)

    remaining = list_backups(backup_dir)
    assert len(remaining) == 2
    assert all(p.suffix == ".sqlite" for p in remaining)


def test_corrupt_backup_does_not_overwrite_target(tmp_path: Path):
    db_path = tmp_path / "memory.sqlite"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val TEXT)")
        conn.execute("INSERT INTO t (val) VALUES ('keep-me')")

    bad_backup = tmp_path / "memory-bad.sqlite.gz"
    with gzip.open(bad_backup, "wb") as f:
        f.write(b"not a valid sqlite file")

    with pytest.raises(Exception):
        restore_sqlite_database(bad_backup, db_path)

    with sqlite3.connect(str(db_path)) as conn:
        val = conn.execute("SELECT val FROM t").fetchone()[0]
    assert val == "keep-me"


def test_cli_backup_and_restore_memory(tmp_path: Path, capsys):
    db_path = tmp_path / "memory.sqlite"
    backup_dir = tmp_path / "cli_backups"
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO items (name) VALUES ('saved')")

    rc = cli_main(
        [
            "backup-memory",
            "--config",
            "config/bot.example.yaml",
            "--db",
            str(db_path),
            "--output-dir",
            str(backup_dir),
            "--retain",
            "3",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    backup_file = Path(out["backup_path"])
    assert backup_file.exists()

    # Wipe DB and restore via CLI
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("DELETE FROM items")

    rc2 = cli_main(
        [
            "restore-memory",
            "--backup",
            str(backup_file),
            "--config",
            "config/bot.example.yaml",
            "--db",
            str(db_path),
        ]
    )
    assert rc2 == 0
    out2 = json.loads(capsys.readouterr().out)
    assert out2["status"] == "ok"

    with sqlite3.connect(str(db_path)) as conn:
        rows = conn.execute("SELECT name FROM items").fetchall()
    assert rows == [("saved",)]
