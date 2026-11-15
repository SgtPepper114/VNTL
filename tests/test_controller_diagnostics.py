"""Controller diagnostic output and exception redaction contracts."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from diagnostics import check_controller


class ControllerDiagnosticsTests(unittest.TestCase):
    def test_invalid_arguments_do_not_initialize_devices(self):
        with patch("diagnostics._check_controller_worker") as worker:
            self.assertEqual(check_controller([]), 2)
            self.assertEqual(check_controller(["a", "b"]), 2)
        worker.assert_not_called()

    def test_success_uses_worker_and_writes_evidence(self):
        caller = threading.get_ident()

        def probe():
            return {"ok": True, "worker_thread": threading.get_ident() != caller}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "result.json"
            with patch("diagnostics._check_controller_worker", side_effect=probe):
                self.assertEqual(check_controller([str(path)]), 0)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"ok": True, "worker_thread": True})

    def test_exception_output_contains_only_type(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with patch("diagnostics._check_controller_worker", side_effect=RuntimeError("private data")):
                self.assertEqual(check_controller([str(path)]), 1)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             {"ok": False, "error_type": "RuntimeError"})


if __name__ == "__main__":
    unittest.main()
