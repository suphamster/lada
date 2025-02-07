<h1 align="center">
  <img src="packaging/flatpak/share/icons/hicolor/128x128/apps/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada
</h1>

## Features
* Recover pixelated adult videos (JAV)
* Watch or export your videos via CLI or GUI

## Usage
### GUI
After opening a file you can either watch a restored version of the provided video in the app (make sure you've enabled the *Preview* toggle) or you can export it to a new file.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshot_gui_1_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/screenshot_gui_1_light.png">
  <img alt="Screenshot showing video preview" src="assets/screenshot_gui_1_dark.png" width="45%">
</picture>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/screenshot_gui_2_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="assets/screenshot_gui_2_light.png">
  <img alt="Screenshot showing video export" src="assets/screenshot_gui_2_dark.png" width="45%">
</picture>

> [!TIP]
> If you've installed the flatpak then it should be available in your regular application launcher. You can also run it via `flatpak run io.github.ladaapp.lada`
> 
> Otherwise, if you've followed the Developer Installation section run the command `lada` to open the app (Make sure you are in the root directory of this project)

> [!NOTE]
> If you've installed Lada from Flathub and drag-and-drop doesn't work then your drag source (your file browser) probably does not support the [File Transfer Portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.FileTransfer.html).
> You can fix/workaround this either by:
>  1) switching or updating your file browser to something that supports it
>  2) giving the app filesystem permissions (e.g. via [Flatseal](https://flathub.org/apps/com.github.tchx84.Flatseal) so it can read the file directly
>  3) use the 'Open' button / file dialog to select the file instead of drag-and-drop

You can find some additional settings in the left sidebar.

### CLI
You can also use the CLI to export the restored video
```shell
lada-cli --input <input video path> --output <output video path>
```
<img src="assets/screenshot_cli_1.png" alt="screenshot showing video export" width="45%">

> [!TIP]
> If you've installed the app via Flathub then the command would look like this (instead of *host* permissions you could also use `--file-forwarding` option)
>  ```shell
>  flatpak run --filesystem=host --command=lada-cli io.github.ladaapp.lada --input <input video path> --output <output video path>
>  ```
> You can also set an alias in your favourite shell and use as the same shorter command as shown above
> ```shell
> alias lada-cli="flatpak run --filesystem=host --command=lada-cli io.github.ladaapp.lada"
>  ```

> [!TIP]
> If you've installed the app via Docker you can pass the parameters via docker run
>  ```shell
> docker run --rm --gpus all --mount type=bind,src=<path to input/output video dir>,dst=/mnt ladaapp/lada:latest --input /mnt/<input video file> --output /mnt/<output video file>
> ```

> [!TIP]
> Lada will write the restored video first to a temporary file before it is being combined with the audio stream from the original file and written to the selected destination.
> Default location is `/tmp`. You can overwrite it by setting the `TMPDIR` environment variable.
> On flatpak you can either pass `--env=TMPDIR=/my/custom/tempdir` to the run command or you can use Flatseal to overwrite this permanently.

You can find out more about additional options by using the `--help` argument.

## Status
Don't expect this to work perfectly, some scenes can be pretty good and close to the real thing. Other scenes can be rather meh and show worse artifacts than the original mosaics.

You'll need a Nvidia (CUDA) GPU and some patience to run the app.
If your GPU is not fast enough to watch the video in real-time you'll have to export it first and watch it later with your favorite media player.
If your card has at least 4-6GB of VRAM then it should work out of the box.

The CPU is used for re-encoding the restored video so shouldn't be too slow either. The app uses a lot of RAM for buffering to increase throughput.
For 1080p content you should be fine with 6-8GB RAM, 4K will need more. This could be lowered by fine-tuning some knowbs in the code if you're that low on RAM.

Technically running the app on your CPU is also supported where *supported* is defined as: It will not crash but processing will be so slow you wish you haven't given it a try.

Here are some speed performance numbers using Lada v0.4.0 on my available hardware to give you an idea what to expect:

| Video name | Video description                                                                                    | Video<br>duration / resolution / FPS | Lada<br>runtime / FPS<br>Nvidia RTX 3050<br>(*Laptop GPU*) | Lada<br>runtime / FPS<br>Nvidia RTX 3090<br>(Desktop GPU) |
|------------|------------------------------------------------------------------------------------------------------|--------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| vid1       | multiple mosaic regions present on all frames                                                        | 1m30s / 10920x1080 / 30 FPS          | 15m33s / 2.8 FPS                                           | 1m41s / 26 FPS                                            |
| vid2       | single mosaic region present on all frames                                                           | 3m0s / 1920x1080 / 30 FPS            | 20m36s / 4.3 FPS                                           | 2m18s / 39 FPS                                            |
| vid3       | half of the video doesn't have any mosaics present,<br>the other half mostly single mosaic per frame | 41m16s / 852x480 / 30 FPS            | 3h20m57s / 6.1 FPS                                         | 13m10s / 94 FPS                                           |


As you can see, Realtime playback for Nvidia RTX 3050 (Laptop GPU) is currently out-of-reach but Preview functionality can still be used to skip through a video (with some loading/buffering) to see what quality to expect from an export.

It may or may not work on Windows and Mac and other GPUs. You'll have to try to follow Developer Installation below and see how far you get.
Patches / reports welcome if you are able to make it run on other systems.

## Installation
On Linux the easiest way to install the app (CLI and GUI) is to get it from Flathub.

<a href='https://flathub.org/apps/details/io.github.ladaapp.lada'><img width='200' alt='Download from Flathub' src='https://flathub.org/api/badge?svg&locale=en'/></a>

> [!CAUTION]
> The flatpak works only with x86_64 CPUs with Nvidia/CUDA GPUs. Make sure your system is using NVIDIAs official driver not `nouveau`.
> (CPU-only inference technically works also, but read the notes in [Status](#Status) first).

The app is also available via docker (CLI only!), you can pull it from dockerhub:
```shell
docker pull ladaapp/lada:latest
````
The image has the same limitations as the flatpak: x86_64 CPU + Nvidia/CUDA GPU only. In order to use your GPU make sure to install
[Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first.

If you don't want to use flatpak/docker, have other hardware specs than what the flatpak is built for or if you're not using Linux you'd need to follow the [Developer installation](#Developer-Installation) steps for now.
Contributions welcome if someone is able to package the app for other systems.

## Models
The project comes with a `generic` mosaic removal / video restoration model that was trained on a diverse set of scenes and is used by default.

> [!TIP]
> For folks currently using or interested in the mosaic restoration model from [DeepMosaics](https://github.com/HypoX64/DeepMosaics):
> It is integrated in Lada and you can use it via CLI or GUI if you prefer. As DeepMosaics is not maintained anymore it's also included in the Flatpak and Docker image of Lada so it's easier to use.

You can select the model to use in the sidepanel or if using the CLI by passing the arguments for path and type of model.

> [!NOTE]
> There are also models for detection for both mosaiced/pixelated and non-obstructed NSFW sources which are used internally for pre-processing and model training.

## Developer Installation

### System dependencies

1) Install Python 3.12
   > If your OS doesn't provide this version you could use `conda` or some of its clones like `micromamba` to install python

2) [Install FFmpeg](https://ffmpeg.org/download.html)

3) [Install GStreamer](https://gstreamer.freedesktop.org/documentation/installing/index.html)

4) [Install GTK](https://www.gtk.org/docs/installations/)

> [!TIP]
> 3/4 are only needed when running the GUI and can be skipped for CLI-only use

### Python dependencies
This is a Python project so let's install our dependencies from PyPi:

1) Create a new virtual environment
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2) [Install PyTorch](https://pytorch.org/get-started/locally)

3) Install this project together with the remaining dependencies
    ```bash
    python -m pip install -e '.[basicvsrpp,gui]'
    ````
    These extras are enough to run the model, GUI and CLI. If you want to train the model(s) or work on the dataset(s) install additional extras `training,dataset-creation`. `gui` is optional and can be skipped if only working with the CLI.

   > [!CAUTION]
   > When installing the dataset-creating extra dependencies `albuminations` will be installed. There seems to be an issue with its dependency management as albumentations will install opencv headless even though opencv is already available and you'll end up with both (you can check via `pip freeze | grep opencv`). 
   > 
   > If you run into conflicts related to OpenCV then uninstall both `opencv-python-headless` and `opencv-python` and install only `opencv-python`. (Noticed on version `albumentations==1.4.24`).

4) Apply patches

    In order to fix resume training of the mosaic restoration model apply the following patch (tested with `mmengine==0.10.6`):
    ```bash
    patch -u ./.venv/lib/python3.12/site-packages/mmengine/runner/loops.py -i patches/adjust_mmengine_resume_dataloader.patch
    ```

    On low-end hardware running mosaic detection model could run into a timeout defined in ultralytics library and the scene would not be restored. The following patch increases this time limit (tested with `ultralytics==8.3.58`):
    ```bash
    patch -u ./.venv/lib/python3.12/site-packages/ultralytics/utils/ops.py patches/increase_mms_time_limit.patch
    ```

### Install models
Download the models from the GitHub Releases page into the `model_weights` directory. The following commands do just that
```shell
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.2.0/lada_mosaic_detection_model_v2.pt'
wget -P model_weights/ -O 'lada_mosaic_restoration_model_generic_v1.2.pth' 'https://github.com/ladaapp/lada/releases/download/v0.5.1-beta3/lada_mosaic_restoration_model_generic_v1.2beta2.pth'
```

If you're interested in running DeepMosaics' restoration model you can also download their pretrained model `clean_youknow_video.pth`
```shell
wget -O model_weights/3rd_party/clean_youknow_video.pth 'https://drive.usercontent.google.com/download?id=1ulct4RhRxQp1v5xwEmUH7xz7AK42Oqlw&export=download&confirm=t'
```

To train the models and create your own datasets you'll also need these files
```shell
wget -P model_weights/3rd_party/ 'https://download.openmmlab.com/mmediting/restorers/basicvsr/spynet_20210409-c6c1bd09.pth'
wget -P model_weights/3rd_party/ 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
wget -P model_weights/3rd_party/ 'https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth'
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.5.1-beta/lada_nsfw_detection_model_v1.3.pt'
wget -P model_weights/ 'https://github.com/ladaapp/lada/releases/download/v0.5.0-beta4/lada_watermark_detection_model_v1.2.pt'
wget -P model_weights/3rd_party/ 'https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt'
```

> [!CAUTION]
> The last download command currently doesn't work as the NudeNet project is set to age-restricted.
> You'll have to be logged into GitHub, then you can download the file manually on their [release page]('https://github.com/notAI-tech/NudeNet/releases/): release `v3.4` / file `640m.pt`
> The model is optional though and only used in `create-mosaic-restoration-dataset.py`.

Now you should be able to run the GUI via `lada` or the CLI via `lada-cli`.


## Training and dataset creation
If you're interested in training your own model(s) or create custom dataset(s) you can find out more in [Training and dataset creation](docs/training_and_dataset_creation.md).

## Credits
This project builds on work done by these fantastic people

* [DeepMosaics](https://github.com/HypoX64/DeepMosaics): Used their code to create mosaic for dataset creation/training, you can also run their clean_youknow_video model in this app. They seem to be the only other open source project trying to solve this task I could find. Kudos to them!
* [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++) / [MMagic](https://github.com/open-mmlab/mmagic): Used as base model for mosaic removal
* [YOLO/Ultralytics](https://github.com/ultralytics/ultralytics): Used as model to detect mosaic regions as well as non-mosaic regions for dataset creation
* [DOVER](https://github.com/VQAssessment/DOVER): Used to assess video quality of created clips during the dataset creation process to filter out low quality videos
* [DNN Watermark / PITA Dataset](https://github.com/tgenlis83/dnn-watermark): Used most of its code for creating the watermark detection dataset used to filter out scenes obstructed with text/watermarks/logos
* [NudeNet](https://github.com/notAI-tech/NudeNet/): Used as an additional NSFW classifier to filter out false positives by our own NSFW segmentation model
* [Twitter Emoji](https://github.com/twitter/twemoji): Used eggplant emoji as base for the app icon (feel free to contribute a better logo)
* PyTorch, FFmpeg, GStreamer, GTK and [all other folks building our ecosystem](https://xkcd.com/2347/)
