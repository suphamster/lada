## Creating `requirements.txt`
```shell
python -m venv .venv_requirements_cli
source ..venv_requirements_cli/bin/activate
pip install pip-tools
pip-compile --extra basicvsrpp -o packaging/requirements-cli.txt setup.py
sed -i 's#opencv-python#opencv-python-headless#' packaging/requirements-cli.txt
pip-compile --extra basicvsrpp,gui -o packaging/requirements-gui.txt setup.py
deactivate
rm -r .venv_requirements_cli
```