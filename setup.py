from setuptools import setup, find_packages
from lada import VERSION

setup(
    name="lada",
    version=str(VERSION),
    description="Remove and recover pixelated areas in adult videos",
    packages=find_packages(where='.',include=['lada','lada.*']),
    install_requires=['torch', 'ultralytics', 'numpy', 'opencv-python', 'tqdm', 'av'],
    extras_require={
        'rvrt': [],
        'tecogan': ['scikit-image'],
        'deepmosaics': ['scikit-image'],
        'basicvsrpp': ['mmengine==0.10.5', 'mmcv'], # mmengine pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
        'gui': ['pycairo', 'PyGObject'],
        'training': ['torchvision', 'albumentations'],
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
