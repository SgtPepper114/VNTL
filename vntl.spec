# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata
from pathlib import Path
import os
from runtime_bootstrap import system_msvc_path

datas = [
    ('hook/vntl_hook_x64.dll', 'hook'),
    ('hook/vntl_hook_x86.dll', 'hook'),
    ('hook/vntl_inject32.exe', 'hook'),
]
binaries = []
hiddenimports = [
    'qasync',
    'mss.windows',
]

for package in ('sudachipy', 'sudachidict_core', 'meikiocr'):
    _d, _b, _h = collect_all(package)
    datas    += _d
    binaries += _b
    hiddenimports += _h

# The bundled onnxruntime hook collects inference/provider DLLs. Its optional
# quantization/training tools are separate applications, not runtime dependencies.
datas += copy_metadata('meikiocr')

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['build_support/runtime_hook.py'],
    excludes=['onnxruntime.quantization', 'onnxruntime.tools', 'onnxruntime.transformers'],
    noarchive=False,
)
# Prefer one compatible C++ runtime over the older copy in Qt's wheel. The
# source installation uses System32; portable builds carry this redistributable.
runtime = system_msvc_path()
if runtime is not None:
    # Qt uses the Windows ICU forwarding APIs. Let the target OS supply them.
    a.binaries = [(dest, source, kind) for dest, source, kind in a.binaries
                  if Path(dest).name.lower() not in ('icuuc.dll', 'icuin.dll')]
    a.binaries = [(dest, str(runtime), kind) if Path(dest).name.lower() == 'msvcp140.dll'
                  else (dest, source, kind) for dest, source, kind in a.binaries]
    if not any(dest.lower() == 'msvcp140.dll' for dest, _, _ in a.binaries):
        a.binaries.append(('msvcp140.dll', str(runtime), 'BINARY'))
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='vntl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=os.environ.get('VNTL_BUILD_CONSOLE') == '1',
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # uac_admin intentionally omitted — most VNs run as normal user so VNTL doesn't
    # need elevation. Running as admin creates a pipe with admin-only security,
    # preventing the normal-user DLL in the game from writing to it.
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='vntl',
)
