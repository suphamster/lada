FROM python:3.12.8-bookworm
RUN useradd --create-home lada
WORKDIR /home/lada
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --no-deps --no-cache-dir --requirement https://raw.githubusercontent.com/ladaapp/lada/23e0fbd1714d733ffffcd23cc04cca0d111b8925/packaging/requirements-cli.txt
RUN pip install --no-deps --no-cache-dir 'lada[basicvsrpp]@https://github.com/ladaapp/lada/archive/refs/tags/v0.4.2.zip#sha1=38d727ecc55146d6fc82045e484db3cced7d2a06'
RUN curl -Ls https://raw.githubusercontent.com/ladaapp/lada/refs/tags/v0.5.0-beta3/patches/increase_mms_time_limit.patch | patch -u -p2 -d /usr/local
RUN mkdir ffmpeg && curl -L 'https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2025-01-15-12-55/ffmpeg-n7.1-152-gd72536008a-linux64-gpl-7.1.tar.xz' | tar xJf - -C ffmpeg  --strip-components 2 ffmpeg-n7.1-152-gd72536008a-linux64-gpl-7.1/bin/ffprobe ffmpeg-n7.1-152-gd72536008a-linux64-gpl-7.1/bin/ffmpeg && chmod -R 555 ffmpeg
ENV PATH="/home/lada/ffmpeg:$PATH"
RUN mkdir -p /home/lada/.config/Ultralytics \
    && echo '{"settings_version":"0.0.6","datasets_dir":"datasets","weights_dir":"weights","runs_dir":"experiments","uuid":"dummy","sync":false,"api_key":"","openai_api_key":"","clearml":false,"comet":false,"dvc":false,"hub":false,"mlflow":false,"neptune":false,"raytune":false,"tensorboard":false,"wandb":false,"vscode_msg":false}' > /home/lada/.config/Ultralytics/settings.json \
    && chown -R lada:lada /home/lada/.config
RUN mkdir model_weights mnt && chmod 555 model_weights && chmod 777 mnt
ADD --checksum=sha256:056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5 \
    https://github.com/ladaapp/lada/releases/download/v0.2.0/lada_mosaic_detection_model_v2.pt model_weights/lada_mosaic_detection_model_v2.pt
ADD --checksum=sha256:6ec6542dde73fbc2086d252a041b41881e3194eaa0bac964348e6f7e8aad007c \
    https://github.com/ladaapp/lada/releases/download/v0.2.1/lada_mosaic_restoration_model_generic_v1.1.pth model_weights/lada_mosaic_restoration_model_generic_v1.1.pth
ADD --checksum=sha256:b79de6fcb1fcafd3ce7c954f4ac788be448ec2d82c6e688aaf18b8ba48fb5b47 \
    https://github.com/ladaapp/lada/releases/download/v0.1.0/lada_mosaic_restoration_model_bj_pov.pth model_weights/lada_mosaic_restoration_model_bj_pov.pth
RUN chmod 444 model_weights/*
USER lada

ENTRYPOINT ["lada-cli"]
CMD ["--help"]