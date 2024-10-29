from setuptools import setup, find_packages

setup(
    name="lada",
    version="0.1.0",
    description="Remove and recover pixelated areas in adult videos",
    packages=find_packages(where='.',include=['lada','lada.*']),
    install_requires=['torch>=2.4.0,<2.5.0', 'ultralytics', 'numpy', 'opencv-python', 'tqdm'],
    extras_require={
        'rvrt': [],
        'tecogan': ['scikit-image'],
        'deepmosaics': ['scikit-image'],
        'basicvsrpp': ['mmengine', 'mmcv', 'mmagic', 'albumentations'],
        'gui': ['pycairo', 'PyGObject'],
        'training': ['av', 'torchvision'],
        'dataset-creation': ['lapx', 'timm', 'einops', 'torchvision'],
        'packaging-flatpak': ['pip-tools', 'req2flatpak']
    },
    include_package_data=True,
    package_data={
        'lada.gui': ['*.css', '*.ui', '*.gresource', '*.gresource.xml']
    },
    entry_points={
        'console_scripts': [
            'lada = lada.gui.main:main',
            'lada-cli = lada.cli.remove_mosaic:cli'
        ],
    }
)
