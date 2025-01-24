## Creating `requirements.txt`
```shell
python -m venv .venv_requirements_cli
source .venv_requirements_cli/bin/activate
pip install pip-tools
pip-compile --extra basicvsrpp -o packaging/requirements-cli.txt setup.py
sed -i 's#opencv-python#opencv-python-headless#' packaging/requirements-cli.txt
pip-compile --extra basicvsrpp,gui -o packaging/requirements-gui.txt setup.py
deactivate
rm -r .venv_requirements_cli
```

## Release Process
1. Create GitHub PR for Flathub/Lada repository (check README there)
2. Once pipeline is green pull this flatpak from flathub
   1. Test GUI: 
        1. Open file
        2. Play/Pause file, Seek file, Change a setting in the sidebar, Reset settings in the sidebar
        3. Export file
   2. Test CLI: export file
3. Assuming all looks good and nothing is broken: Push a commit in Ladaapp/lada with these changes:
    1. Bump version in `lada/__init__.py`
    2. Write short release notes in `packaging/flatpak/share/metainfo/io.github.ladaapp.lada.metainfo.xml`
    3. update used git tags in `packaging/docker/Dockerfile.Release ` to this new version
4. Create a Release on Ladaapp/Lada for this commit. Release notes should be mostly copy-paste from what was written in the xml file but can be more verbose if necessary.
5. Update the PR in Flathub/lada repo with the release git tag and commit id
6. Once pipeline went through merge the PR on Flathub/lada. After a few hours it should be available on flathub
7. Build and test this docker image locally
8. Push the new image to Dockerhub (make sure to use the same tag as was used on git, as well as 'latest')
9. Add `-dev` suffix to version in `lada/__init__.py`