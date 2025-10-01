<h1 align="center">
  <img src="packaging/flatpak/share/icons/hicolor/128x128/apps/io.github.ladaapp.lada.png" alt="Lada Icon" style="display: block; width: 64px; height: 64px;">
  <br>
  Lada
</h1>

*Lada* is a tool designed to recover pixelated adult videos (JAV). It helps restore the visual quality of such content, making it more enjoyable to watch.

## Features

- **Recover Pixelated Videos**: Restore pixelated or mosaic scenes in adult videos.
- **Watch/Export Videos**: Use either the CLI or GUI to watch or export your restored videos.

## Usage

### GUI

After opening a file, you can either watch the restored version of the provided video within the app (make sure you've enabled the *Preview* toggle) or export it to a new file.

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

Additional settings can be found in the left sidebar.

### CLI

You can also use the command-line interface (CLI) to export restored videos:

```shell
lada-cli --input <input video path>
```
<img src="assets/screenshot_cli_1.png" alt="screenshot showing video export" width="45%">

For more information about additional options, use the `--help` argument.

> [!TIP]
> Lada writes the restored video to a temporary file before combining it with the audio stream from the original file and saving it to the selected destination.
> You can overwrite [the default location](https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir) by setting the `TMPDIR` environment variable to another location of you choice.

## Restoration options

Lada utilizes specialized models for the two main steps of the processing pipeline: Detection and Restoration. You can choose different models for each task.

**Mosaic Restoration Models:**

*   **basicvsrpp-v1.2 (Default)** A general-purpose model trained on diverse video scenes. Delivers mostly good results.
*   **deepmosaics:** Restoration model from the project [DeepMosaics](https://github.com/HypoX64/DeepMosaics). Worse quality than basicvsrpp-v1.2.

> [!NOTE]
> The DeepMosaics model should be worse in most/all scenarios. It’s integrated because the DeepMosaics project is not maintained anymore, and I wanted to provide an easy way to try it out and
compare.

**Mosaic Detection Models:**

*   **v3.1-fast (Default):** Fast and efficient.
*   **v3.1-accurate:**  More accurate than v3.1-fast, but slower. Not always better than v2.
*   **v2:** Slowest of all but often provides better mosaic detection than v3.1-accurate.

You configure the models in the side panel, or when using the CLI by specifying path and type of the model as arguments.

## Status
Don't expect this to work perfectly, some scenes can be pretty good and close to the real thing. Other scenes can be rather meh and show worse artifacts than the original mosaics.

You'll need a Nvidia (CUDA) GPU and some patience to run the app. If your card has at least 4-6GB of VRAM then it should work out of the box.

The CPU is used for encoding the restored video so shouldn't be too slow either (current version does not ship with GPU video encoders). You can adjust encoder options to your needs on the CLI.

The app also needs quite a bit of RAM for buffering to increase throughput. For 1080p content you should be fine with 6-8GB RAM, 4K will need a lot more.

If you want to use watch the restored video in the GUI preview mode in real-time you'll need a pretty beefy machine or you'll see the player pausing until next restored frames are computed.
GUI Preview mode will need less CPU resources as it will not have to encode the video but will use more additional RAM for buffering.

If your GPU is not fast enough to watch the video in real-time you'll have to export it first and watch it later with your favorite media player.

Technically running the app on your CPU is also supported where *supported* is defined as: It will not crash but processing will be so slow you wish you haven't given it a try.

Here are some speed performance numbers using Lada v0.7.0 on my available hardware to give you an idea what to expect (used libx264 codec with default settings; RTX 3090 results are limited by CPU encoding and could be faster with other settings):

| Video name | Video description                                                                                    | Video<br>duration / resolution / FPS | Lada<br>runtime / FPS<br>Nvidia RTX 3050<br>(*Laptop GPU*) | Lada<br>runtime / FPS<br>Nvidia RTX 3090<br>(Desktop GPU) |
|------------|------------------------------------------------------------------------------------------------------|--------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| vid1       | multiple mosaic regions present on all frames                                                        | 1m30s / 10920x1080 / 30 FPS          | 3m36s / 12 FPS                                             | 1m33s / 30 FPS                                            |
| vid2       | single mosaic region present on all frames                                                           | 3m0s / 1920x1080 / 30 FPS            | 4m11s / 21 FPS                                             | 2m16s / 39 FPS                                            |
| vid3       | half of the video doesn't have any mosaics present,<br>the other half mostly single mosaic per frame | 41m16s / 852x480 / 30 FPS            | 26m30s / 46 FPS                                            | 10m20s / 119 FPS                                          |


It may or may not work on Windows and Mac and other GPUs. You'll have to try to follow [Build](#build) below and see how far you get.

There have been successful reports of installing Lada on Windows with Nvidia and Intel GPUs. AMD GPUs should also work but probably not with Windows as PyTorch/ROCm builds are only available for Linux.

Patches / reports welcome if you are able to make it run on other systems or have a suggestion how to improve the documentation.

## Installation
### Using Flatpak
The easiest way to install the app (CLI and GUI) on Linux is via Flathub:

<a href='https://flathub.org/apps/details/io.github.ladaapp.lada'><img width='200' alt='Download from Flathub' src='https://flathub.org/api/badge?svg&locale=en'/></a>

> [!NOTE]
> The Flatpak version works only with x86_64 CPUs and Nvidia/CUDA GPUs. Ensure your system NVIDIA GPU drivers are upt-to-date.
> It can also be used without a GPU but it will be very slow.

> [!TIP]
> After installation you should find Lada in your application launcher to start the GUI. You can also run it via `flatpak run io.github.ladaapp.lada`.

> [!TIP]
> When using the CLI via Flatpak we need to make the file/directory available by giving it permission to the file system
>  ```shell
>  flatpak run --filesystem=host --command=lada-cli io.github.ladaapp.lada --input <input video path>
>  ```
> You may want to set an alias to make it easier to use
> ```shell
> alias lada-cli="flatpak run --filesystem=host --command=lada-cli io.github.ladaapp.lada"
>  ```
> You could also give the filesystem permission permanently via [Flatseal](https://flathub.org/apps/com.github.tchx84.Flatseal) 

> [!TIP]
> If you installed Lada from Flathub and drag-and-drop doesn't work, your file browser might not support [File Transfer Portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.FileTransfer.html).
> You can fix this by:
>  1) Switching or updating your file browser to one that supports it.
>  2) Granting the app filesystem permissions (e.g., via [Flatseal](https://flathub.org/apps/com.github.tchx84.Flatseal) so it can read files directly).
>  3)  Using the 'Open' button to select the file instead of drag-and-drop.

### Using Docker

The app is also available via Docker (CLI only). You can pull it from Docker Hub with the following command:

```shell
docker pull ladaapp/lada:latest
````

> [!TIP]
> When using the Docker you'll need to make the file/directory available to the Container as well as the GPU:
>  ```shell
> docker run --rm --gpus all --mount type=bind,src=<input video path>,dst=/mnt ladaapp/lada:latest --input "/mnt/<input video file>"
> ```

> [!NOTE]
> The Flatpak version works only with x86_64 CPUs and Nvidia/CUDA GPUs. Ensure your system NVIDIA GPU drivers are upt-to-date.
> It can also be used without a GPU but it will be very slow.

### Alternative Installation Methods

If you prefer not to use Flatpak or Docker, have different hardware specifications, or are using a non-Linux system, follow the [Build](#build) steps.
Contributions are welcome if someone can package the app for other systems!

There is some work-in-progress to package for [Windows](https://github.com/ladaapp/lada/issues/77).

## Build
If you want to start hacking on this project you'll need to install the app from source. Check out the detailed installation guides for [Linux](docs/linux_install.md) and [Windows](docs/windows_install.md).

## Training and dataset creation
For instructions on training your own models and datasets, refer to [Training and dataset creation](docs/training_and_dataset_creation.md).

## Credits
This project builds upon work done by these fantastic individuals and projects:

* [DeepMosaics](https://github.com/HypoX64/DeepMosaics): Provided code for mosaic creation during dataset creation/training. Also inspired me to start this project.
* [BasicVSR++](https://ckkelvinchan.github.io/projects/BasicVSR++) / [MMagic](https://github.com/open-mmlab/mmagic): Used as the base model for mosaic removal.
* [YOLO/Ultralytics](https://github.com/ultralytics/ultralytics): Used for training mosaic and NSFW detection models.
* [DOVER](https://github.com/VQAssessment/DOVER):  Used to assess video quality of created clips during the dataset creation process to filter out low-quality clips.
* [DNN Watermark / PITA Dataset](https://github.com/tgenlis83/dnn-watermark): Used most of its code for creating a watermark detection dataset used to filter out scenes obstructed by text/watermarks/logos.
* [NudeNet](https://github.com/notAI-tech/NudeNet/): Used as an additional NSFW classifier to filter out false positives by our own NSFW segmentation model
* [Twitter Emoji](https://github.com/twitter/twemoji): Provided eggplant emoji as base for the app icon.
* [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN): Used their image degradation model design for our mosaic detection model degradation pipeline.
* PyTorch, FFmpeg, GStreamer, GTK and [all other folks building our ecosystem](https://xkcd.com/2347/)
