# Training
The app uses two models: mosaic detection and mosaic restoration.
The goal of the mosaic detection model is to detect for each frame of the video if and where pixelated/mosaic regions exist.
It will try to crop and cut small clips and hand them over to the mosaic restoration model. This will try to recover what it can from those degraded frames and come up with a somewhat plausible replacement for those images.
These restored clip will replace the original content when they're reassembled with the original frames.

There is also a NSFW detection model used only to create the datasets for the other two models.

The following sections describe how to train and create a dataset for each model.
If you're not interested in training specific models you can use the pretrained model weights from Lada where needed.

> [!NOTE]
> Lada models were trained using Python 3.12 / Torch 2.4.1-cuda-12.4 / MMCV 2.2.0 / Ultralytics 8.3.23 but I would suggest to use latest versions as described in the Developer Installation section.

> [!TIP]
> To gather source material for your dataset [yt-dlp](https://github.com/yt-dlp/yt-dlp) and [gallery-dl](https://github.com/mikf/gallery-dl) are your friends.
> 
> To find and remove duplicate files I can recommend the tools [Czkawka](https://github.com/qarmin/czkawka) and [cbird](https://github.com/scrubbbbs/cbird).

> [!TIP]
> In some scripts OpenCV QT-based GUI features are using for debugging purposes. 
> OpenCV currently does not support wayland, you may want to force QT to use its X11 backend by setting the following environment variable
> `export QT_QPA_PLATFORM=xcb`.

## Mosaic restoration model
Before we can train the model we'll need to create a dataset.
AFAIK, there are no publicly available datasets for such purpose and I'll not provide one either. But you can create your own dataset for training mosaic removal models with the following procedure:

```shell
python scripts/dataset_creation/create-mosaic-restoration-dataset.py --input <input dir> --output-root <output dir>
```
Here `<input dir>` should be a directory containing your source material (adult video files without mosaics).

The script will detect regions of NSFW material and cut and crop short clips to be used for mosaic restoration training.

The script doesn't necessarily need to be called with additional options, but you should check them out (`--help`) and play with it to understand what's going on before running this on a lot of data.

For example:
You can optimize worker and memory limits according to your machine. You can also run the script in parallel on different subset of data using different GPUs.
There are options to create mosaic clips as well which can be useful to inspect generated mosaic clips.
Depending on your source material use the `--stride-length` option to prevent sampling too many scenes from the same (long) files.

Additional metadata and filtering can be adjusted as well. Check-out the *filter* / *add-metadata* switches.

Try it on a small subset of your data first to see how it works.
Also, check out the code `MosaicVideoDataset` in `mosaic_video_dataset.py` as well as the dataloader/dataset settings in `mosaic_restoration_generic_stage{1,2}.py` in the `config` dir to understand how this generated dataset will be used in training.

With the default settings only cropped NSFW scenes and their segmentation masks will be generated (two directories). Mosaics will gen generated on-the-fly while training by torch dataset `MosaicVideoDataset`.
There are options though to generate full-size scenes as well as mosaic scenes. I would not suggest to enable these options on your final large-sized dataset as it will take a lot of time and storage space.

The script uses the NSFW detection model. Some filtering options are available in the training config files and in the script for automatic filtering of invalid data.
Neither of those are perfectly accurate, and you probably want to validate and remove false-positive clips manually.

> [!TIP]
> I used the following workflow for manual clean-up:
> (1) open the directory of created NSFW scenes in your file explorer in thumbnail view. (2) Wait until thumbnails have been created. (3) Based on what you can see in the thumbnail delete files if they contain watermarks or don't look like an actual NSFW scene. (4) write a shell script to delete corresponding mask and json metadata files 

Now, with a dataset at hand we're ready to train a model.

Training the mosaic restoration model is done in two steps. You can find training scripts in the project root directory and related configuration files in the `config` directory.
The first stage consists of training a BasicVSR++ model only with pixel loss (you'll need to create a dataset first)
```shell
python scripts/training/train-mosaic-restoration-basicvsrpp.py configs/basicvsrpp/mosaic_restoration_generic_stage1.py
```
> You can continue an interrupted run by adding `--resume` to the command line.

Before we can continue training stage2 you'll have to convert the trained weights into the GAN-compatible model with the following script
```shell
python scripts/training/convert-weights-basicvsrpp-stage1-to-stage2.py
```
Now we can continue training with additional GAN and perceptual losses.
```shell
python scripts/training/train-mosaic-restoration-basicvsrpp.py configs/basicvsrpp/mosaic_restoration_generic_stage2.py --load-from experiments/basicvsrpp/mosaic_restoration_generic_stage1/iter_10000_converted.pth
```

If you're happy with the model you can export it for inference and remove the discriminator model via:
```shell
python scripts/training/export-weights-basicvsrpp-stage2-for-inference.py
```

Note that the model is implemented in the MMagic / MMEngine framework. If you need to adjust model or training parameters you can do that by adjusting
the files in the `config` directory.

I'd recommend to read through [MMengine documentation](https://mmengine.readthedocs.io/en/latest/) first if you're not familiar with that library.

> [!NOTE]
> Besides this BasicVSR++-based mosaic restoration model you can find training scripts for other/previously used models (like DeepMosaics).
> I would not recommend working on them anymore but kept them mostly to show how to integrate other restoration/super-resolution models into Lada in case you stumble upon another interesting model.

## NSFW detection model
The purpose of this dataset is to train an image segmentation model which we can feed video frames to detect if and where NSFW regions exist in the image (binary segmentation task).
This model is then only used in the mosaic restoration and mosaic detection dataset creation scripts.
Creating this dataset is a labor-intensive process, you'll have to hand-label each image and draw segmentation masks around all NSFW regions visible in each image.

I used and recommend the tool [labelme](https://github.com/wkentaro/labelme) for this task.

Setup and use another virtual environment to avoid dependency conflicts with Lada dependencies when installing labelme
```shell
python -m venv .venv_labelme
source .venv_labelme/bin/activate
pip install labelme
```

Create training and validation directories for your NSFW detection dataset and fill it with diverse set of NSFW video frames and images
```shell
mkdir -p datasets/nsfw_detection_labelme/{train,val}
```

Then start the tool with this command

```shell
labelme --flags sfw --labels nsfw --nodata --autosave datasets/nsfw_detection_labelme/train
```

I'd suggest to keep the masks relatively tight around each object. In practise mosaics are often not very precise and also cover a good amount of SFW content around NSFW parts.
We'll randomly extend those masks automatically when working on mosaic detection or mosaic restoration so we should label them relatively precisely in this step.

In *labelme* use the *Draw Polygon* tool to label NSFW regions in the image.

<img src="assets/screenshot_labelme_nsfw.png" alt="screenshot showing LabelMe program using polygon tool to label NSFW regions" height="300px">

You should also add some images without any visible NSFW content. For those, create a *SFW* label and no Polygons.

<img src="assets/screenshot_labelme_sfw.png" alt="screenshot showing LabelMe program labeling image as SFW" height="300px">

Then continue this process with a couple more images.

Now let's train the model

> [!NOTE]
> To continue switch back to lada virtual environment `.venv`. Use the `.venv_labelme` only when annotating files in labelme.

We're training a YOLO v11 segmentation model (`yolo11m-seg`).
YOLO or rather the library we're using to train it (*ultralytics*) does not support labelme format so we'll have to convert into the format they can understand:
```shell
mkdir -p datasets/nsfw_detection/{train,val}/{images,labels}
python scripts/dataset_creation/convert-dataset-labelme-to-yolo.py --dir-in datasets/nsfw_detection_labelme/train --dir-out-images datasets/nsfw_detection/train/images --dir-out-labels datasets/nsfw_detection/train/labels
python scripts/dataset_creation/convert-dataset-labelme-to-yolo.py --dir-in datasets/nsfw_detection_labelme/val --dir-out-images datasets/nsfw_detection/val/images --dir-out-labels datasets/nsfw_detection/val/labels
```

With that step out of the way we're now ready to start the training process. Simply run
```shell
python scripts/training/train-nsfw-detection-yolo.py
```

> [!TIP]
> Under the hood, `ultralytics` package is used to train the model. You can read their [documentation](https://docs.ultralytics.com/) for further details.


Once that is done test it on some real-world NSFW videos (not from the source material you've been training with) using the following script:
```shell
python scripts/evaluation/view-yolo.py --input <path to your nsfw file> --model-path experiments/yolo/segment/train_nsfw_detection_yolo11m/weights/best.pt --screenshot-dir datasets/nsfw_detection_labelme/train
```

This will open a very simple GUI where you can seek through the video frame-by-frame to check the detection result (masks and prediction confidence levels).
If you find frames with false-positives or other parts the model incorrectly classified hit the `S` key to take a screenshot and save the frame in the specified directory.

<img src="assets/screenshot_view_yolo.png" alt="screenshot showing view-yolo.py tool" width="45%">

Then you can fire up labelme once more and annotate this file properly.

Repeat these steps to annotate, convert, train, validate a couple of times, and you have built yourself a NSFW detection model.

> [!TIP]
> After the first training round you can run the model on some other files, convert the predictions from YOLO to labelme and validate and correct them in labelme before adding them to the dataset

## Mosaic detection model
Purpose of the mosaic detection model is to detect if and where mosaic regions are in a given video frame. It's used as the first part of the restoration pipeline of Lada.
Assuming you have a directory of NSFW videos and a trained NSFW detection model (either the pretrained model weights from lada or you've trained your own version following above-mentioned procedure)
the creation of this dataset will be an automatic process.

You can create a dataset for mosaic detection using the following command:

```shell
mosaic_detection_dataset_source_material="datasets/mosaic_detection_raw"
mosaic_detection_dataset_tmp_dir="datasets/mosaic_detection_tmp"

python create-mosaic-detection-dataset.py --input-root "$mosaic_detection_dataset_source_material"  --output-root "$mosaic_detection_dataset_tmp_dir"
```

The script will run the NSFW detection model on the source material. For frames with NSFW content it will use the segmentation mask and the original video frame
to create a new image where the NSFW region is replaced with a mosaic / pixelize pattern. This image as well as the segmentation mask of the mosaic is then saved in the output root directory.

The script comes with some options you may want to check and adjust before you're running this on a huge dataset.

> [!TIP]
> You can re-use the source material used for mosaic restoration or NSFW detection dataset creation.


The mosaic detection model uses the same architecture as the NSFW detection model YOLO `yolo11m-seg`. Similarly to the steps above we need to convert the dataset first
to a format that the training library of YOLO *ultralytics* understands. For this we'll convert the binary mask images (PNG files) first to COCO format and then to YOLO format.

```shell
python scripts/dataset_creation/convert-dataset-mask-images-to-coco.py --images-dir "$mosaic_detection_dataset_tmp_dir/img" --masks-dir "$mosaic_detection_dataset_tmp_dir/mask" --output-file "$mosaic_detection_dataset_tmp_dir/coco.json"
python scripts/dataset_creation/convert-dataset-coco-to-yolo.py --coco-file "$mosaic_detection_dataset_tmp_dir/coco.json" --yolo-labels-dir "$mosaic_detection_dataset_tmp_dir/labels"
```

Now we just need to split it into a training and test set.

Now we just need to split it into a training and test set.
The following shell commands will do just that. You may want to adjust the variable `val_samples_count`.
It will randomly select items from the dataset up the given number and move it the validation directory `val`.
Everything else will land in the training set `train`.

```shell
mosaic_detection_dataset_dir="datasets/mosaic_detection"
val_samples_count=500

# extract $val_samples_count samples as validation dataset
mkdir -p "$mosaic_detection_dataset_dir"/{train,val} "$mosaic_detection_dataset_dir"/val/{labels,images}
find $mosaic_detection_dataset_tmp_dir/img -type f | sort -R | head -n $val_samples_count | while read img_path ; do 
    mask_path="$mosaic_detection_dataset_tmp_dir/labels/$(basename ${img_path%.jpg}.txt)"
    mv "$img_path" "$mosaic_detection_dataset_dir/val/images"
    mv "$mask_path" "$mosaic_detection_dataset_dir/val/labels"
done
# whats left will be used as training dataset
mv "$mosaic_detection_dataset_tmp_dir/img" "$mosaic_detection_dataset_dir/train/images"
mv "$mosaic_detection_dataset_tmp_dir/labels" "$mosaic_detection_dataset_dir/train/labels"

# delete tmp dataset
# rm -r $mosaic_detection_dataset_tmp_dir
```

Now we can train the model via
```shell
python scripts/training/train-mosaic-detection-yolo.py
```

> [!TIP]
> Under the hood, `ultralytics` package is used to train the model. You can read their [documentation](https://docs.ultralytics.com/) for further details.

To check its performance you can use the `view-yolo.py` script as described in the training section of the NSFW detection model.

## Watermark detection model
This model is used to filter out scenes detected by the NSFW model obstructed by watermarks, text or logos. Otherwise, we could introduce unwanted artifacts when
applying mosaics on them when added to the mosaic restoration dataset.

Luckily for us there exists a public dataset for this! [PITA Dataset](https://huggingface.co/datasets/bastienp/visible-watermark-pita)
They also provide it in YOLO format so you guessed it, we're training another YOLOv11 model. This time the detection not segmentation variant but it's the same process.

Download the val and train YOLO zip files and extract them to `datasets/watermark_detection/{train,val}`

> [!NOTE]
> The dataset differentiates two classes 'text' and 'logo'. For our purpose we don't need to know what kind of watermark we're dealing with

In order to train the model run
```shell
python scripts/training/train-watermark-detection-yolo.py
```
