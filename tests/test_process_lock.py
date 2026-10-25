from __future__ import annotations

import asyncio
from pathlib import Path

from biometric_service import process_lock


def test_watchlist_lock_creates_lock_file_and_releases(tmp_path: Path) -> None:
    async def exercise() -> None:
        fallback = asyncio.Lock()
        async with process_lock.watchlist_file_lock(
            tmp_path, shared=True, fallback_lock=fallback
        ):
            assert (tmp_path / ".watchlist-access.lock").exists()
        async with process_lock.watchlist_file_lock(
            tmp_path, shared=False, fallback_lock=fallback
        ):
            assert (tmp_path / ".watchlist-access.lock").exists()

    asyncio.run(exercise())


def test_windows_fallback_serializes_access(tmp_path: Path, monkeypatch) -> None:
    async def exercise() -> None:
        monkeypatch.setattr(process_lock, "fcntl", None)
        fallback = asyncio.Lock()
        order: list[str] = []
        entered = asyncio.Event()

        async def first() -> None:
            async with process_lock.watchlist_file_lock(
                tmp_path, shared=True, fallback_lock=fallback
            ):
                order.append("first-enter")
                entered.set()
                await asyncio.sleep(0.02)
                order.append("first-exit")

        async def second() -> None:
            await entered.wait()
            async with process_lock.watchlist_file_lock(
                tmp_path, shared=False, fallback_lock=fallback
            ):
                order.append("second-enter")

        await asyncio.gather(first(), second())
        assert order == ["first-enter", "first-exit", "second-enter"]

    asyncio.run(exercise())
