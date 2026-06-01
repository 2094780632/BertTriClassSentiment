"""
PyInstaller 运行时 hook：在一切 import 之前修复 PyTorch DLL 加载路径。
"""
import os
import sys


def _patch_torch_dll_path():
    """将 torch/lib 目录加入 DLL 搜索路径（必须在 import torch 之前执行）。"""
    base = getattr(sys, '_MEIPASS', None)
    if not base:
        return

    # 是否 onedir 模式（EXE 在子目录 _internal 旁边）
    internal = os.path.join(base, '_internal')

    candidates = [
        os.path.join(internal, 'torch', 'lib'),
        os.path.join(base, 'torch', 'lib'),
        os.path.join(base, '_internal', 'torch', 'lib'),
    ]

    print(f"[runtime_hook] _MEIPASS={base}", file=sys.stderr)
    print(f"[runtime_hook] candidates={candidates}", file=sys.stderr)

    for path in candidates:
        if os.path.isdir(path):
            print(f"[runtime_hook] Found torch lib: {path}", file=sys.stderr)
            # os.add_dll_directory 修改 DLL 搜索顺序（Python ≥3.8）
            if hasattr(os, 'add_dll_directory'):
                try:
                    os.add_dll_directory(path)
                except OSError as e:
                    print(f"[runtime_hook] add_dll_directory failed: {e}", file=sys.stderr)
            # PATH 兜底
            os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
            # 也设置 CONDA_DLL_SEARCH_MODIFICATION_ENABLE 以允许 conda 环境的 DLL 搜索（如果有）
            os.environ.setdefault('CONDA_DLL_SEARCH_MODIFICATION_ENABLE', '1')


_patch_torch_dll_path()

