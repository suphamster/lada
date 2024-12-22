python -m venv .venv_labelme
source .venv_labelme/bin/activate
pip install labelme

mkdir -p datasets/nsfw_detection_labelme/{train,val}

labelme --flags sfw --labels nsfw --nodata --autosave datasets/nsfw_detection_labelme/train
labelme --flags sfw --labels nsfw --nodata --autosave datasets/nsfw_detection_labelme/val
