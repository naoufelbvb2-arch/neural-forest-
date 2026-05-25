"""SQLiteMemoryStore: persistent storage for conversations, projects, preferences."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteMemoryStore:
    """Local SQLite-backed memory store that persists between conversations.

    Stores conversations, projects, and user preferences on disk.
    Contents are injected into the system prompt before each forward pass —
    no gradient updates needed.

    Args:
        db_path: Path to the SQLite database file.

    TODO: implement in PROMPT 2
    """

    def __init__(self, db_path: str | Path = "memory.db") -> None:
        self.db_path = Path(db_path)
        # TODO: implement schema creation, CRUD methods

    def save(self, key: str, value: str, category: str = "general") -> None:
        """Persist a key-value memory entry.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("SQLiteMemoryStore.save — implement in PROMPT 2")

    def load(self, category: str | None = None) -> list[dict]:
        """Retrieve memory entries, optionally filtered by category.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("SQLiteMemoryStore.load — implement in PROMPT 2")

    def to_prompt_text(self) -> str:
        """Format stored memories as a system prompt prefix.

        TODO: implement in PROMPT 2
        """
        raise NotImplementedError("SQLiteMemoryStore.to_prompt_text — implement in PROMPT 2")
