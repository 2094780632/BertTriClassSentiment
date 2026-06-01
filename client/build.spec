# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path

_base = Path(SPECPATH)        # ← 改为 SPECPATH

a = Analysis(
    ['main.py'],
    pathex=[str(_base)],
    binaries=[],
    datas=[
        ('form.ui', '.'),
        ('icon.ico', '.'),
        (str(_base.parent / 'models'), 'models'),
    ],
    hiddenimports=[
        'transformers', 'torch', 'sklearn', 'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'PyQt5', 'PyQt5.uic',
        'scripts.config', 'scripts.model',
    ],
    excludes=['tkinter', 'IPython', 'jupyter'],
    hookspath=[],
    runtime_hooks=[],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz, a.scripts, a.binaries, a.zipfiles, a.datas,
    [],
    name='TriClassSentiment',
    debug=False,
    strip=False,
    upx=True,
    console=False,
    icon=str(_base / 'icon.ico'),
)