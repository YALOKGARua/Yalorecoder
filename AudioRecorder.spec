# -*- mode: python ; coding: utf-8 -*-
import os
import sys

from PyInstaller.utils.hooks import collect_all

py_dir = os.path.dirname(sys.executable)

sf_datas, sf_binaries, sf_hiddenimports = collect_all('_soundfile_data')

a = Analysis(
    ['recorder.py'],
    pathex=[],
    binaries=[
        (os.path.join(py_dir, 'vcruntime140.dll'), '.'),
        (os.path.join(py_dir, 'vcruntime140_1.dll'), '.'),
    ] + sf_binaries,
    datas=[
        ('icon.ico', '.'),
    ] + sf_datas,
    hiddenimports=['soundfile', 'pystray._win32'] + sf_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

import customtkinter
ctk_path = os.path.dirname(customtkinter.__file__)
a.datas += Tree(ctk_path, prefix='customtkinter')

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AudioRecorder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    uac_admin=True,
    icon=['icon.ico'],
)
