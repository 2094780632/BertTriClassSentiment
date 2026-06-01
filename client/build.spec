# -*- mode: python ; coding: utf-8 -*-
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs, copy_metadata

_base = Path(SPECPATH)

# ——— 收集 torch 的 DLL ———
_torch_binaries = []
_torch_datas = []

_VC_DLLS = ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'msvcp140_1.dll',
            'msvcp140_2.dll', 'vcruntime140d.dll', 'msvcp140d.dll', 'msvcp140_atomic_wait.dll',
            'concrt140.dll']

def _add_dir_as_binaries(src_dir, dst_dir):
    """递归收集 src_dir 下所有文件到 binaries 列表"""
    import glob as _glob
    for f in _glob.glob(str(src_dir / '**' / '*'), recursive=True):
        fp = Path(f)
        if fp.is_file():
            rel = fp.relative_to(src_dir)
            _torch_binaries.append((str(fp), str(Path(dst_dir) / rel.parent)))

try:
    import torch
    _torch_lib = Path(torch.__file__).parent / 'lib'
    if _torch_lib.exists():
        _torch_binaries.append((str(_torch_lib / 'c10.dll'), 'torch/lib'))
        _torch_binaries.append((str(_torch_lib / 'torch.dll'), 'torch/lib'))
        _torch_binaries.append((str(_torch_lib / 'torch_cpu.dll'), 'torch/lib'))
        _torch_binaries.append((str(_torch_lib / 'torch_python.dll'), 'torch/lib'))
        for dll in sorted(_torch_lib.glob('*.dll')):
            _torch_binaries.append((str(dll), 'torch/lib'))
except Exception:
    pass

# ——— VC++ 运行时 ———
_vc_candidates = [
    Path(sys.base_prefix) / 'DLLs',
    Path(sys.prefix) / 'DLLs',
    Path(sys.base_prefix),
    Path(sys.prefix),
    Path(r'C:\Windows\System32'),
    Path(r'C:\Windows\SysWOW64'),
]

_need_vc = {'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll'}
for _d in _vc_candidates:
    if _d.exists():
        for _vc_name in _need_vc:
            _vc_file = _d / _vc_name
            if _vc_file.exists() and not any(b[0] == str(_vc_file) for b in _torch_binaries):
                _torch_binaries.append((str(_vc_file), '.'))

# ——— transformers / huggingface 元数据 ———
_hf_metadata = []
try:
    for pkg in ['transformers', 'tokenizers', 'huggingface_hub']:
        try:
            _hf_metadata.extend(copy_metadata(pkg))
        except Exception:
            pass
except Exception:
    pass

a = Analysis(
    ['main.py'],
    pathex=[str(_base), str(_base.parent)],
    binaries=_torch_binaries,
    datas=[
        ('form.ui', '.'),
        ('icon.ico', '.'),
    ] + _hf_metadata,
    hiddenimports=[
        'transformers', 'torch', 'sklearn', 'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'PyQt5', 'PyQt5.uic',
        'scripts',
        'scripts.config',
        'scripts.model',
        # 修复 DLL 加载所需的额外隐式导入
        'torch._C', 'torch._C._nn', 'torch._C._VariableFunctions',
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
    upx=True,
    console=False,
    icon=str(_base / 'icon.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='TriClassSentiment',
)