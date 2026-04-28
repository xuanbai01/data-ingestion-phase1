"""Shared pytest fixtures."""
import os
import tempfile

import pytest

from database import db


@pytest.fixture
def temp_db(monkeypatch):
    """
    Point db.DB_PATH at an isolated temp file for the duration of a test.
    Tables are created up front so individual tests don't need to.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    monkeypatch.setattr(db, "DB_PATH", path)
    db.create_tables()
    try:
        yield path
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
