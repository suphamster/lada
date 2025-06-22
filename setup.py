from setuptools import setup, find_packages
from lada import VERSION

setup(
    name="lada",
    version=str(VERSION),
    description="Remove and recover pixelated areas in adult videos",
    python_requires='>=3.12',
    packages=find_packages(where='.',include=['lada','lada.*']),
    # ultralytics: pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
    # av: The latest version (14.4.0 at time of writing) ships with nvenc (nvidia hardware encoder support) but does not ship libx265 (no software h265/hevc encoder).
    #     Version 13.1.0 has support for libx265 but no nvenc. It is fixed upstream but no release yet. If you want to use both codecs atm you'll have to built av package yourself.
    install_requires=['torch', 'ultralytics==8.3.157', 'numpy', 'opencv-python', 'tqdm', 'av==13.1.0'],
    extras_require={
        'deepmosaics': ['scikit-image'],
        'basicvsrpp': ['mmengine==0.10.7', 'torchvision'], # mmengine pinned as we apply a custom patch. When upstream releases a new version, check if we can remove the patch
        'gui': ['pycairo', 'PyGObject'],
        'gui-dev': ['pygobject-stubs'],
        'training': ['torchvision', 'albumentations', 'tensorboard', 'standard-imghdr'],
        'dataset-creation': ['lap>=0.5.12', 'timm', 'einops', 'torchvision', 'pillow']
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
