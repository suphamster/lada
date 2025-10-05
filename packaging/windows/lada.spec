# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from os.path import join as ospj
import shutil
import os
import pathlib

GVSBUILD_DIR = "C:/project/build"
GTK_RELEASE_DIR = os.path.join(GVSBUILD_DIR, "gtk/x64/release")
LADA_BASE_DIR = str(pathlib.Path(__name__).parent.parent.parent.resolve())

datas = collect_data_files('lada.gui', includes=['**/*.ui'])
datas += [
        (ospj(LADA_BASE_DIR, 'lada/gui/style.css'), '.'),
        (ospj(LADA_BASE_DIR, 'lada/gui/resources.gresource'), '.'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_detection_model_v2.pt'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_detection_model_v3.1_accurate.pt'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_detection_model_v3.1_fast.pt'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/lada_mosaic_restoration_model_generic_v1.2.pth'), 'model_weights'),
        (ospj(LADA_BASE_DIR, 'model_weights/3rd_party/clean_youknow_video.pth'), 'model_weights/3rd_party'),
]

bin_ffmpeg = shutil.which("ffmpeg.exe")
assert bin_ffmpeg is not None
bin_ffprobe = shutil.which("ffprobe.exe")
assert bin_ffprobe is not None
binaries = [
	(bin_ffmpeg, "bin"),
	(bin_ffprobe, "bin"),
	(ospj(GTK_RELEASE_DIR, "bin/gdbus.exe"), "."),
	# The following fixes runtime warning at startup (AFAIK did not break anything)
	# (process:11556): GLib-GIRepository-CRITICAL **: 11:22:58.188: Unable to load platform-specific GIO introspection data: Typelib file for namespace 'GioWin32' (any version) not found
	(ospj(GTK_RELEASE_DIR, "lib/girepository-1.0/GioWin32-2.0.typelib"), "gi_typelibs"),
]

a = Analysis(
    [ospj(LADA_BASE_DIR, 'lada/gui/main.py')],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={
        "gi": {
            "icons": ["Adwaita"],
            "themes": ["Adwaita"],
            "languages": [],
            "module-versions": {
                "Gtk": "4.0",
            },
        },
    },
    runtime_hooks=[ospj(LADA_BASE_DIR, "packaging/windows/pyinstaller_runtime_hook_lada.py")],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='lada',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=[ospj(LADA_BASE_DIR, 'packaging/flatpak/share/icons/hicolor/128x128/apps/io.github.ladaapp.lada.png')],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='lada',
)
