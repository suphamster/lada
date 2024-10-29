```shell
mkdir -p datasets/nsfw_detection/{train,val}/{images,labels}
python convert-labelme-to-yolo.py --dir-in datasets/nsfw_detection/train --dir-out-images datasets/nsfw_detection/train/images --dir-out-labels datasets/nsfw_detection/train/labels
python convert-labelme-to-yolo.py --dir-in datasets/nsfw_detection/val --dir-out-images datasets/nsfw_detection/val/images --dir-out-labels datasets/nsfw_detection/val/labels

python train-yolo-nsfw-detection.py
```
