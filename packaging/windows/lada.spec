# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['lada\\gui\\main.py'],
    pathex=[],
    binaries=[
        ('C:\\Users\\test\\AppData\\Local\\Microsoft\\WinGet\\Links\\ffmpeg.exe', 'bin'),
        ('C:\\Users\\test\\AppData\\Local\\Microsoft\\WinGet\\Links\\ffprobe.exe', 'bin'),
        ('C:/project/build/gtk/x64/release/bin/gdbus.exe', '.')
    ],
    datas=[
        ('lada/gui/*.css', '.'),
        ('lada/gui/*.ui', 'lada/gui'),
        ('lada/gui/preview/*.ui', 'lada/gui/preview'),
        ('lada/gui/export/*.ui', 'lada/gui/export'),
        ('lada/gui/fileselection/*.ui', 'lada/gui/fileselection'),
        ('lada/gui/config/*.ui', 'lada/gui/config'),
        ('lada/gui/resources.gresource', '.'),
        ('model_weights/lada_mosaic_detection_model_v2.pt', 'model_weights'),
        ('model_weights/lada_mosaic_detection_model_v3.1_accurate.pt', 'model_weights'),
        ('model_weights/lada_mosaic_detection_model_v3.1_fast.pt', 'model_weights'),
        ('model_weights/lada_mosaic_restoration_model_generic_v1.2.pth', 'model_weights'),
        ('model_weights/3rd_party/clean_youknow_video.pth', 'model_weights/3rd_party')],
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
    runtime_hooks=["pyinstaller_runtime_hook_lada.py"],
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
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['packaging\\flatpak\\share\\icons\\hicolor\\128x128\\apps\\io.github.ladaapp.lada.png'],
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
