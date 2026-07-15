"""Sqlite checkpoint storage for multi-turn agent conversations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DIR = Path(".agent_db")
CHECKPOINT_DB = CHECKPOINT_DIR / "checkpoints.sqlite"


def get_checkpointer() -> SqliteSaver:
    """Return a SqliteSaver backed by the project checkpoint database."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    return SqliteSaver(conn)
