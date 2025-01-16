from setuptools import setup, find_packages
from lada import VERSION

setup(
    name="lada",
    version=str(VERSION),
    description="Remove and recover pixelated areas in adult videos",
    packages=find_packages(where='.',include=['lada','lada.*']),
    # ultralytics pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
    # todo: pin av down to 13.1.0. With av >=14.0.0 lada gets stuck on export for some reason. needs to be investigated
    install_requires=['torch', 'ultralytics==8.3.58', 'numpy', 'opencv-python', 'tqdm', 'av==13.1.0'], 
    extras_require={
        'rvrt': [],
        'tecogan': ['scikit-image'],
        'deepmosaics': ['scikit-image'],
        'basicvsrpp': ['mmengine==0.10.5', 'mmcv'], # mmengine pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
        'gui': ['pycairo', 'PyGObject'],
        'gui-dev': ['pygobject-stubs'],
        'training': ['torchvision', 'albumentations', 'tensorboard'],
        'dataset-creation': ['lap>=0.5.12', 'timm', 'einops', 'torchvision']
    },
    include_package_data=True,
    package_data={
        'lada.gui': ['*.css', '*.ui', '*.gresource', '*.gresource.xml']
    },
    entry_points={
        'console_scripts': [
            'lada = lada.gui.main:main',
            'lada-cli = lada.cli.main:main'
        ],
    }
)
