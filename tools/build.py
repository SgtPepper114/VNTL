"""Build VNTL with the same Windows CRT bootstrap in all analysis processes."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
environment = os.environ.copy()
bootstrap = str(root / "build_support")
environment["PYTHONPATH"] = os.pathsep.join(filter(None, (bootstrap, environment.get("PYTHONPATH"))))
if sys.platform == "win32":
    # Native tools such as Poppler may add incompatible namesakes (icuuc.dll)
    # to a developer's PATH. Resolve wheel dependencies plus Windows libraries,
    # rather than accidentally bundling unrelated desktop-tool DLLs.
    windows_dir = Path(os.environ["SystemRoot"])
    directories = (Path(sys.executable).parent, Path(sys.base_prefix),
                   Path(sys.base_prefix) / "DLLs", windows_dir / "System32", windows_dir)
    environment["PATH"] = os.pathsep.join(dict.fromkeys(map(str, directories)))
raise SystemExit(subprocess.call(
    [sys.executable, "-m", "PyInstaller", "--noconfirm", str(root / "vntl.spec")],
    cwd=root, env=environment,
))
