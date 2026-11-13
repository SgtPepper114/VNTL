"""Run before PyInstaller's Qt runtime hook changes Windows DLL directories."""
from runtime_bootstrap import preload_windows_runtime

preload_windows_runtime()
