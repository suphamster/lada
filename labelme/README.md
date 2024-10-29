python -m venv .venv
source .venv/bin/activate
pip install labelme

QT_QPA_PLATFORM=xcb labelme --flags p datasets/nsfw_detection/train --nodata

