"""Cross-process lock for shared watchlist files used by biometric workers.

On Linux, flock allows concurrent readers while excluding readers during enrollment
and deletion writes. Windows local tests use an asyncio-lock fallback; they do not
validate Linux multi-process locking semantics.
"""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    fcntl = None  # type: ignore[assignment]


@asynccontextmanager
async def watchlist_file_lock(
    data_dir: Path | str,
    *,
    shared: bool,
    fallback_lock: asyncio.Lock,
) -> AsyncIterator[None]:
    """Lock watchlist files across worker processes on Linux.

    ``shared=True`` permits concurrent searches. ``shared=False`` excludes searches
    while enrollment, deletion, or collection initialization mutates persisted state.
    """
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".watchlist-access.lock"
    # Create the file on every platform; Windows still relies on the local fallback.
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)

    if fcntl is None:
        os.close(fd)
        async with fallback_lock:
            yield
        return

    operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
    acquired = False
    try:
        # Never block the event loop while another process holds the lock.
        await asyncio.to_thread(fcntl.flock, fd, operation)
        acquired = True
        yield
    finally:
        try:
            if acquired:
                await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
