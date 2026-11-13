"""Build-only bootstrap inherited by PyInstaller's isolated analysis children."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_bootstrap import preload_windows_runtime

preload_windows_runtime()
