# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(100000)

from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata

_base = Path(SPECPATH)

# ——— transformers / huggingface 元数据 ———
_hf_metadata = []
for pkg in ['transformers', 'tokenizers', 'huggingface_hub']:
    try:
        _hf_metadata.extend(copy_metadata(pkg))
    except Exception:
        pass

a = Analysis(
    ['main.py'],
    pathex=[str(_base), str(_base.parent)],
    binaries=[],   # 让 PyInstaller 自动收集所有 DLL
    datas=[
        ('form.ui', '.'),
        ('icon.ico', '.'),
    ] + _hf_metadata,
    hiddenimports=[  # 删除了不存在的 _nn 和 _VariableFunctions
        'transformers', 'torch', 'sklearn', 'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'PyQt5', 'PyQt5.uic',
        'scripts', 'scripts.config', 'scripts.model',
        'torch.utils', 'torch.utils.cpp_extension',
        'torch.utils.collect_env',
        'transformers.models.bert',
        'transformers.models.bert.modeling_bert',
        'tokenizers', 'tokenizers.decoders',
        'torchvision', 'PIL', 'PIL.Image',
        'numpy.core._multiarray_umath',
        'numpy.random._common',
        'numpy.linalg._umath_linalg',
    ],
    excludes=['tkinter', 'IPython', 'jupyter'],
    hookspath=[],
    runtime_hooks=[str(_base / 'runtime_hook.py')],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TriClassSentiment',
    debug=False,
    strip=False,
    upx=False,          # 先禁用 UPX
    console=False,
    icon=str(_base / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,          # 先禁用 UPX
    upx_exclude=[],
    name='TriClassSentiment',
)