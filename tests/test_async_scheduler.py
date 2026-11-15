"""Real Qt/qasync scheduling without network, game windows or user settings."""
from __future__ import annotations

import asyncio
import contextvars
import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QObject, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication

from async_scheduler import create_event_loop


class AsyncSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.loop = create_event_loop(self.app)
        asyncio.set_event_loop(self.loop)
        self.addCleanup(self._close)
        self.gui_thread = threading.get_ident()

    def _close(self):
        if not self.loop.is_closed():
            self.loop.close()
        asyncio.set_event_loop(None)

    def run_async(self, coroutine):
        # Independent native-timer watchdog makes regressions fail rather than
        # leaving the test runner hung on the scheduler being tested.
        expired = []
        watchdog = QTimer()
        watchdog.setSingleShot(True)
        watchdog.setTimerType(Qt.TimerType.PreciseTimer)
        watchdog.timeout.connect(lambda: (expired.append(True), self.loop.stop()))
        watchdog.start(5000)
        try:
            result = self.loop.run_until_complete(coroutine)
        finally:
            watchdog.stop()
        self.assertFalse(expired, "event-loop watchdog expired")
        return result

    def test_repeated_sleep_and_call_soon_keep_waking_without_qt_timers(self):
        async def scenario():
            for _ in range(100):
                await asyncio.sleep(0.001)
                future = self.loop.create_future()
                self.loop.call_soon(future.set_result, threading.get_ident())
                self.assertEqual(await future, self.gui_thread)
            self.assertEqual(self.loop._timer.findChildren(QTimer), [])
        self.run_async(scenario())

    def test_call_soon_is_queued_not_reentrant_and_preserves_fifo(self):
        order = []
        async def scenario():
            done = self.loop.create_future()
            self.loop.call_soon(lambda: (order.append("first"), self.loop.call_soon(order.append, "nested")))
            self.loop.call_soon(order.append, "second")
            self.assertEqual(order, [])
            self.loop.call_later(0.02, done.set_result, None)
            await done
        self.run_async(scenario())
        self.assertEqual(order, ["first", "second", "nested"])

    def test_delays_use_monotonic_deadlines_and_never_fire_early(self):
        async def scenario():
            started = self.loop.time()
            future = self.loop.create_future()
            deadline = started + 0.04
            self.loop.call_at(deadline, lambda: future.set_result(self.loop.time()))
            fired = await future
            self.assertGreaterEqual(fired, deadline)
            self.assertLess(fired - deadline, 1.0)
        self.run_async(scenario())

    def test_new_earlier_deadline_interrupts_background_wait(self):
        async def scenario():
            later = self.loop.call_later(float("inf"), lambda: self.fail("far future callback ran"))
            future = self.loop.create_future()
            self.loop.call_later(0.01, future.set_result, "ready")
            self.assertEqual(await future, "ready")
            later.cancel()
        self.run_async(scenario())

    def test_cancelled_handle_and_sleep_and_wait_for_timeout(self):
        called = []
        async def scenario():
            self.loop.call_soon(called.append, "cancelled soon").cancel()
            self.loop.call_later(0.02, called.append, "cancelled later").cancel()
            task = self.loop.create_task(asyncio.sleep(3600))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            with self.assertRaises(TimeoutError):
                await asyncio.wait_for(asyncio.Event().wait(), 0.01)
            await asyncio.sleep(0.03)
        self.run_async(scenario())
        self.assertEqual(called, [])

    def test_threadsafe_callback_keeps_worker_context_and_runs_on_gui_thread(self):
        value = contextvars.ContextVar("scheduler_test", default="unset")
        async def scenario():
            future = self.loop.create_future()
            def worker():
                value.set("worker")
                self.loop.call_soon_threadsafe(
                    lambda: future.set_result((value.get(), threading.get_ident())))
            thread = threading.Thread(target=worker)
            thread.start()
            observed = await future
            thread.join()
            self.assertEqual(observed, ("worker", self.gui_thread))
        self.run_async(scenario())

    def test_explicit_context_and_threadsafe_cancellation(self):
        value = contextvars.ContextVar("scheduler_explicit", default="unset")
        context = contextvars.copy_context()
        context.run(value.set, "chosen")
        called = []
        cancelled = self.loop.call_soon_threadsafe(called.append, "cancelled")
        cancelled.cancel()
        async def scenario():
            future = self.loop.create_future()
            self.loop.call_soon_threadsafe(lambda: future.set_result(value.get()), context=context)
            self.assertEqual(await future, "chosen")
        self.run_async(scenario())
        self.assertEqual(called, [])

    def test_executor_future_completes_back_on_gui_thread(self):
        async def scenario():
            results = await asyncio.gather(*[
                asyncio.to_thread(lambda: threading.get_ident()) for _ in range(30)])
            self.assertTrue(all(identity != self.gui_thread for identity in results))
            self.assertEqual(threading.get_ident(), self.gui_thread)
        self.run_async(scenario())

    def test_callback_exception_reports_to_loop_and_future_work_continues(self):
        errors = []
        self.loop.set_exception_handler(lambda loop, context: errors.append(context))
        def failure():
            raise ValueError("expected callback failure")
        async def scenario():
            self.loop.call_soon(failure)
            await asyncio.sleep(0.01)
        self.run_async(scenario())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0]["exception"], ValueError)

    def test_loop_can_run_again_and_pauses_callbacks_between_runs(self):
        called = []
        self.run_async(asyncio.sleep(0))
        self.loop.call_soon(called.append, "second run")
        self.app.processEvents()
        self.assertEqual(called, [])
        self.run_async(asyncio.sleep(0.01))
        self.assertEqual(called, ["second run"])

    def test_close_cancels_pending_callbacks_and_stops_driver(self):
        called = []
        self.loop.call_soon(called.append, "soon")
        self.loop.call_later(3600, called.append, "later")
        scheduler = self.loop._timer
        self.loop.close()
        self.app.processEvents()
        self.assertEqual(called, [])
        self.assertFalse(scheduler._thread.is_alive())
        self.assertEqual(scheduler._heap, [])
        with self.assertRaises(RuntimeError):
            self.loop.call_soon(called.append, "closed")
        with self.assertRaises(RuntimeError):
            self.loop.call_soon_threadsafe(called.append, "closed")

    def test_nested_qt_processing_cannot_reenter_asyncio_callbacks(self):
        order = []
        def outer():
            self.loop.call_soon(order.append, "inner")
            self.app.processEvents()
            order.append("outer end")
        async def scenario():
            self.loop.call_soon(outer)
            await asyncio.sleep(0.02)
        self.run_async(scenario())
        self.assertEqual(order, ["outer end", "inner"])

    def test_large_ready_batch_yields_to_queued_qt_ui_events(self):
        class UIWake(QObject):
            ping = pyqtSignal()
        wake = UIWake()
        completed, observed = [], []
        wake.ping.connect(lambda: observed.append(len(completed)), Qt.ConnectionType.QueuedConnection)
        async def scenario():
            self.loop.call_soon(wake.ping.emit)
            for i in range(1000):
                self.loop.call_soon(completed.append, i)
            await asyncio.sleep(0.05)
        self.run_async(scenario())
        self.assertEqual(len(completed), 1000)
        self.assertEqual(len(observed), 1)
        self.assertLess(observed[0], 1000)

    def test_debug_mode_preserves_current_handle_and_cleans_it_after_errors(self):
        self.loop.set_debug(True)
        current = []
        async def scenario():
            future = self.loop.create_future()
            def callback():
                current.append(self.loop._current_handle is not None)
                future.set_result(None)
            self.loop.call_soon(callback)
            await future
        self.run_async(scenario())
        self.assertEqual(current, [True])
        self.assertIsNone(self.loop._current_handle)


if __name__ == "__main__":
    unittest.main()
