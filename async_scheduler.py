"""qasync integration with monotonic deadlines independent of native Qt timers.

A single condition-waiting thread delivers queued Qt wakeups. All asyncio
callbacks still execute on the GUI thread, through their original Handles.
This avoids allocating a Windows timer ID for every coroutine resumption.
"""
from __future__ import annotations

import asyncio
import heapq
import logging
import math
import threading
import time

import qasync
from PyQt6.QtCore import QObject, Qt, pyqtSignal, pyqtSlot

logger = logging.getLogger(__name__)


class _DeadlineScheduler(QObject):
    _wake_requested = pyqtSignal()
    _MAX_CALLBACKS = 64
    _BATCH_SECONDS = 0.004

    def __init__(self, loop):
        super().__init__()
        self._loop = loop
        self._condition = threading.Condition()
        self._heap = []
        self._sequence = 0
        self._stopped = False
        self._active = False
        self._wake_pending = False
        self._dispatching = False
        self._debug = False
        # Explicit QueuedConnection also prevents synchronous reentry when a
        # callback schedules another callback on this same GUI thread.
        self._wake_requested.connect(self._dispatch, Qt.ConnectionType.QueuedConnection)
        self._thread = threading.Thread(target=self._wait_for_deadline,
                                        name="vntl-async-deadlines", daemon=True)
        self._thread.start()

    def add_callback(self, handle, delay=0):
        delay = float(delay)
        if math.isnan(delay):
            raise ValueError("delay must not be NaN")
        deadline = self._loop.time() + max(0.0, delay)
        with self._condition:
            if self._stopped:
                raise RuntimeError("Event loop is closed")
            self._sequence += 1
            heapq.heappush(self._heap, (deadline, self._sequence, handle))
            self._condition.notify()
        return handle

    def resume(self):
        with self._condition:
            if self._stopped:
                raise RuntimeError("Event loop is closed")
            self._active = True
            self._condition.notify()

    def pause(self):
        with self._condition:
            self._active = False
            self._wake_pending = False
            self._condition.notify()

    def _discard_cancelled_head(self):
        # The condition is held by every caller. Callback execution never holds
        # it, so user callbacks may enqueue new work or close the loop safely.
        while self._heap and self._heap[0][2]._cancelled:
            heapq.heappop(self._heap)

    def _wait_for_deadline(self):
        while True:
            with self._condition:
                while True:
                    if self._stopped:
                        return
                    self._discard_cancelled_head()
                    if not self._active or self._wake_pending or not self._heap:
                        self._condition.wait()
                        continue
                    remaining = self._heap[0][0] - self._loop.time()
                    if remaining > 0:
                        # Avoid platform timeout overflow for very distant or
                        # infinite deadlines; newly scheduled work wakes us.
                        self._condition.wait(min(remaining, 3600.0))
                        continue
                    self._wake_pending = True
                    break
            # Emitting a queued signal is the worker's only interaction with
            # Qt. It never runs a Handle or accesses a widget.
            self._wake_requested.emit()

    @pyqtSlot()
    def _dispatch(self):
        with self._condition:
            if self._stopped or not self._active:
                self._wake_pending = False
                self._condition.notify()
                return
            if self._dispatching:
                return
            self._dispatching = True
            # A callback scheduled by this batch belongs to a later Qt turn.
            sequence_limit = self._sequence
            due_before = self._loop.time()
        started = time.monotonic()
        try:
            for _ in range(self._MAX_CALLBACKS):
                with self._condition:
                    self._discard_cancelled_head()
                    if (self._stopped or not self._active or not self._heap
                            or self._heap[0][0] > due_before
                            or self._heap[0][1] > sequence_limit):
                        break
                    _, _, handle = heapq.heappop(self._heap)
                if not handle._cancelled:
                    self._run_handle(handle)
                handle = None
                if (not self._loop.is_running()
                        or time.monotonic() - started >= self._BATCH_SECONDS):
                    break
        finally:
            with self._condition:
                self._dispatching = False
                self._wake_pending = False
                self._condition.notify()

    def _run_handle(self, handle):
        started = time.monotonic()
        if self._debug:
            self._loop._current_handle = handle
        try:
            # Handle._run preserves contextvars and routes ordinary exceptions
            # through loop.call_exception_handler, exactly as asyncio does.
            handle._run()
        except (KeyboardInterrupt, SystemExit) as exc:
            # Raising out of a PyQt slot can abort the interpreter. Re-raise at
            # run_forever's Python boundary after the Qt event loop unwinds.
            self._loop._fatal_callback_error = exc
            self._loop.stop()
        finally:
            if self._debug:
                self._loop._current_handle = None
                elapsed = time.monotonic() - started
                if elapsed >= self._loop.slow_callback_duration:
                    logger.warning("Async callback took %.3f seconds", elapsed)

    def set_debug(self, enabled):
        self._debug = bool(enabled)

    def stop(self):
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._active = False
            for _, _, handle in self._heap:
                handle.cancel()
            self._heap.clear()
            self._condition.notify()
        self._thread.join(timeout=1.0)


class StableQEventLoop(qasync.QEventLoop):
    """Keep qasync's Qt/IO integration, replacing only its callback scheduler."""

    def __init__(self, app):
        super().__init__(app)
        previous = self._timer
        # qasync 0.28 constructs its timer before configuring the base loop.
        # No user callbacks exist at this point; replace it before first use.
        if getattr(previous, "_SimpleTimer__callbacks", None):
            raise RuntimeError("Unexpected qasync callbacks during loop construction")
        previous.stop()
        previous.deleteLater()
        self._fatal_callback_error = None
        self._timer = _DeadlineScheduler(self)
        self._timer.set_debug(self.get_debug())

    def call_soon_threadsafe(self, callback, *args, context=None):
        self._check_closed()
        if asyncio.iscoroutinefunction(callback):
            raise TypeError("coroutines cannot be used with call_soon_threadsafe")
        if not callable(callback):
            raise TypeError("callback must be callable")
        handle = asyncio.Handle(callback, args, self, context=context)
        return self._timer.add_callback(handle)

    def run_forever(self):
        self._check_closed()
        if self.is_running():
            raise RuntimeError("Event loop already running")
        self._timer.resume()
        try:
            result = super().run_forever()
        finally:
            self._timer.pause()
        if self._fatal_callback_error is not None:
            error, self._fatal_callback_error = self._fatal_callback_error, None
            raise error
        return result


def create_event_loop(app):
    """Create the GUI event loop before scheduling application coroutines."""
    return StableQEventLoop(app)
