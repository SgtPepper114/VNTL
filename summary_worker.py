"""Run one background summary at a time, independently of subtitle rendering."""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class BackgroundSummarizer:
    def __init__(self, translator, context, on_committed: Callable[[], None],
                 check_ready: Callable[[], None] | None = None):
        self._translator = translator
        self._context = context
        self._on_committed = on_committed
        self._check_ready = check_ready
        self._task: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def request(self, *, force: bool = False) -> bool:
        """Schedule work now; never wait for it or queue duplicate paid calls."""
        if self._closed or self.is_running:
            return False
        if not force and not self._context.needs_summarization():
            return False
        if self._context.prepare_summarization() is None:
            return False
        task = asyncio.create_task(self._run(), name="vntl-background-summary")
        self._task = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run(self) -> None:
        task = asyncio.current_task()
        try:
            while not self._closed:
                if self._check_ready is not None:
                    self._check_ready()
                committed = await self._translator._summarize_context()
                if not committed:
                    # Keep all original memory on failure/staleness. Retry only
                    # after a later successful translation or manual request.
                    break
                self._on_committed()
                # New lines may have reached the threshold while we waited.
                if not self._context.needs_summarization():
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Background summary failed (%s); keeping available memory", type(exc).__name__)
        finally:
            # Cancellation may already have started work for a new context.
            if self._task is task:
                self._task = None

    def cancel(self) -> None:
        """Invalidate pending work when settings or the loaded route changes."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()

    def stop(self) -> None:
        self._closed = True
        for task in tuple(self._tasks):
            if not task.done():
                task.cancel()

    async def aclose(self) -> None:
        self.stop()
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)
