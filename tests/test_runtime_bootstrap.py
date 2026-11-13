import os
from pathlib import Path
import subprocess
import sys
import unittest


class NativeRuntimeTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "Windows DLL loading regression")
    def test_qt_then_onnx_import_succeeds_with_production_bootstrap(self):
        # A fresh process matters: other tests may already have imported NumPy,
        # Qt or ORT and conceal the old-CRT/native-access-violation regression.
        environment = os.environ.copy()
        environment["QT_QPA_PLATFORM"] = "offscreen"
        child = subprocess.run([
            sys.executable, "-c",
            "from runtime_bootstrap import preload_windows_runtime; "
            "assert preload_windows_runtime(); "
            "from PyQt6.QtWidgets import QApplication; app=QApplication([]); "
            "import onnxruntime; assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()",
        ], cwd=Path(__file__).resolve().parents[1], env=environment,
            capture_output=True, timeout=30)
        self.assertEqual(child.returncode, 0, child.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
