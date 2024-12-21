import argparse
import json
import pathlib
from concurrent import futures
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import sleep

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from lada.lib import VideoMetadata, Mask, mask_utils
from lada.lib import video_utils, image_utils
from lada.lib.degradation_utils import MosaicRandomDegradationParams, apply_frame_degradation
from lada.dover.evaluate import VideoQualityEvaluator
from lada.lib.image_utils import pad_image
from lada.lib.mosaic_utils import get_random_parameter, addmosaic_base, get_mosaic_block_size, get_mosaic_block_size_v2
from lada.lib.nsfw_scene_detector import SceneGenerator, Scene, CroppedScene, NsfwFramesGenerator
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry

disable_ultralytics_telemetry()

@dataclass
class SceneProcessingOptions:
    output_dir: Path
    save_flat: bool
    out_size: int
    save_cropped: bool
    save_uncropped: bool
    resize_crops: bool
    preserve_crops: bool
    save_mosaic: bool
    degrade_mosaic: bool
    save_as_images: bool

@dataclass
class FileProcessingOptions:
    scene_min_length: int
    scene_max_length: int
    stride_length: int

class SceneProcessingData:
    def __init__(self):
        self.images = []
        self.mask_images = []
        self.pads = []
        self.mosaic_images = []
        self.mosaic_mask_images = []
        self.mosaic_pads = []
        self.meta = {}
        self.quality_score = None


def get_base_mosaic_block_size(scene: Scene):
    box_sizes = [(r - l + 1) * (b - t + 1) for t, l, b, r in scene.get_boxes()]
    median_idx = np.argsort(box_sizes)[len(box_sizes) // 2]
    _, mask_image_representative, box = scene[median_idx]
    t, l, b, r = box
    filter = np.ones_like(mask_image_representative, dtype=bool)
    filter[t:b + 1, l:r + 1, :] = False
    mask_image_representative[filter] = 0

    # not sure what we'll use later, lets save all variants for now

    mosaic_block_size = dict(
        mosaic_size_v2=get_mosaic_block_size_v2(mask_image_representative),
        mosaic_size_v1_normal=get_mosaic_block_size(mask_image_representative, 'normal'),
        mosaic_size_v1_bounding=get_mosaic_block_size(mask_image_representative, 'bounding'))
    return mosaic_block_size


class MosaicRandomParams:
    def __init__(self, scene: Scene):
        box_sizes = [(r-l+1) * (b-t+1)for t,l,b,r in scene.get_boxes()]
        median_idx = np.argsort(box_sizes)[len(box_sizes) // 2]
        _, mask_image_representative, box = scene[median_idx]
        t, l, b, r = box
        filter = np.ones_like(mask_image_representative, dtype=bool)
        filter[t:b + 1, l:r + 1,:] = False
        if sum(mask_image_representative[filter]) > 0:
            print(f"non zero pixels outside of box: {sum(mask_image_representative[filter])}")
        mask_image_representative[filter] = 0
        self.mosaic_size, self.mosaic_mod, self.mosaic_rectangle_ratio, self.mosaic_feather_size = get_random_parameter(mask_image_representative)
        self.mosaic_mask_dilation_iterations = np.random.choice(range(2))

class SceneProcessingDataContainer:
    def __init__(self, mosaic_params: MosaicRandomParams | None,
                 mosaic_degradation_params: MosaicRandomDegradationParams | None,
                 uncropped: SceneProcessingData | None,
                 cropped_scaled: SceneProcessingData | None,
                 cropped_unscaled: SceneProcessingData | None):
        self.mosaic_degradation_params = mosaic_degradation_params
        self.uncropped = uncropped
        self.cropped_scaled = cropped_scaled
        self.cropped_unscaled = cropped_unscaled
        self.mosaic_params = mosaic_params


def save_vid(frame_dir: pathlib.Path, fileprefix: str, imgs: list[np.ndarray[np.uint8]], fps=30, gray=False):
    frame_dir.mkdir(parents=True, exist_ok=True)
    if gray:
        video_utils.write_masks_to_video_file(imgs, str(frame_dir.joinpath(f"{fileprefix}.mkv").absolute()), fps)
    else:
        video_utils.write_frames_to_video_file(imgs, str(frame_dir.joinpath(f"{fileprefix}.mp4").absolute()), fps)

def save_meta(frame_dir: pathlib.Path, fileprefix: str, meta: dict):
    frame_dir.mkdir(parents=True, exist_ok=True)
    with open(str(frame_dir.joinpath(f"{fileprefix}.json").absolute()), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False)

def save_imgs(frame_dir: pathlib.Path, fileprefix: str, file_ext: str, imgs: np.ndarray,
              jpeg_quality_level=95):
    frame_dir.mkdir(parents=True, exist_ok=True)
    for i in range(len(imgs)):
        path = str(frame_dir.joinpath(f"{fileprefix}{i:06d}{file_ext}").absolute())
        try:
            if file_ext == ".jpg":
                cv2.imwrite(path, imgs[i], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_level])
            else:
                cv2.imwrite(path, imgs[i])
                cv2.imwrite(path, imgs[i])
        except Exception as e:
            print(e)


def get_dir_and_file_prefix(output_dir, name, file_name, scene_id, save_flat=False, file_suffix="-", video=False):
    if save_flat:
        file_prefix = f"{file_name}-{scene_id:06d}{file_suffix}"
        frame_dir = output_dir.joinpath(name)
    else:
        if video:
            file_prefix = f"{scene_id:06d}"
            frame_dir = output_dir.joinpath(name).joinpath(file_name)
        else:
            file_prefix = ""
            frame_dir = output_dir.joinpath(name).joinpath(file_name).joinpath(f"{scene_id:06d}")
    return frame_dir, file_prefix


def process_cropped_scene(cropped_scene: CroppedScene, scene: Scene, scene_processing_options: SceneProcessingOptions, data: SceneProcessingDataContainer, scene_max_height, scene_max_width, start_idx=0, end_exclusive_idx=None):
    if end_exclusive_idx is None:
        end_exclusive_idx = len(cropped_scene)
    for i, (cropped_image, cropped_mask_image, cropped_box) in enumerate(cropped_scene[start_idx:end_exclusive_idx], start=start_idx):
        if scene_processing_options.save_mosaic:
            cropped_mask_mosaic = mask_utils.dilate_mask(cropped_mask_image, iterations=data.mosaic_params.mosaic_mask_dilation_iterations)
            cropped_mosaic_image, cropped_mask_mosaic = addmosaic_base(cropped_image, cropped_mask_mosaic, data.mosaic_params.mosaic_size,
                                                                       model=data.mosaic_params.mosaic_mod, rect_ratio=data.mosaic_params.mosaic_rectangle_ratio,
                                                                       feather=data.mosaic_params.mosaic_feather_size)
            if scene_processing_options.save_cropped:
                if scene_processing_options.resize_crops:
                    resized_rectangle_mask_mosaic_image = image_utils.resize(cropped_mask_mosaic, scene_processing_options.out_size,
                                                                 interpolation=cv2.INTER_NEAREST)
                    if scene_processing_options.degrade_mosaic:
                        resize_me = apply_frame_degradation(cropped_mosaic_image, data.mosaic_degradation_params)
                    else:
                        resize_me = cropped_mosaic_image
                    resized_rectangle_mosaic_image = image_utils.resize(resize_me, scene_processing_options.out_size,
                                                            interpolation=cv2.INTER_CUBIC)
                    final_mask_mosaic_image, _ = pad_image(resized_rectangle_mask_mosaic_image, scene_processing_options.out_size, scene_processing_options.out_size)
                    final_mosaic_image, final_mosaic_image_pad = pad_image(resized_rectangle_mosaic_image, scene_processing_options.out_size, scene_processing_options.out_size)
                    data.cropped_scaled.mosaic_images.append(final_mosaic_image)
                    data.cropped_scaled.mosaic_mask_images.append(final_mask_mosaic_image)
                    data.cropped_scaled.mosaic_pads.append(final_mosaic_image_pad)
                if scene_processing_options.preserve_crops:
                    final_mask_mosaic_image, _ = pad_image(cropped_mask_mosaic, scene_max_height, scene_max_width, mode='zero')
                    if scene_processing_options.degrade_mosaic:
                        pad_me = apply_frame_degradation(cropped_mosaic_image, data.mosaic_degradation_params)
                    else:
                        pad_me = cropped_mosaic_image
                    final_mosaic_image, final_mosaic_image_pad = pad_image(pad_me, scene_max_height, scene_max_width, mode='zero')
                    data.cropped_unscaled.mosaic_images.append(final_mosaic_image)
                    data.cropped_unscaled.mosaic_mask_images.append(final_mask_mosaic_image)
                    data.cropped_unscaled.mosaic_pads.append(final_mosaic_image_pad)

            if scene_processing_options.save_uncropped:
                scene_image, scene_mask_image, _ = scene[i]
                mosaic_image = scene_image.copy()
                t, l, b, r = cropped_box
                mosaic_image[t:b + 1, l:r + 1, :] = cropped_mosaic_image
                if scene_processing_options.degrade_mosaic:
                    mosaic_image = apply_frame_degradation(mosaic_image, data.mosaic_degradation_params)
                mosaic_mask_image = np.zeros_like(scene_mask_image, dtype=cropped_mask_mosaic.dtype)
                mosaic_mask_image[t:b + 1, l:r + 1] = cropped_mask_mosaic
                data.uncropped.mosaic_images.append(mosaic_image)
                data.uncropped.mosaic_mask_images.append(mosaic_mask_image)

        if scene_processing_options.save_cropped:
            if scene_processing_options.resize_crops:
                resized_rectangle_image = image_utils.resize(cropped_image, scene_processing_options.out_size,
                                                 interpolation=cv2.INTER_CUBIC)
                resized_rectangle_mask_image = image_utils.resize(cropped_mask_image, scene_processing_options.out_size,
                                                      interpolation=cv2.INTER_NEAREST)
                final_image, final_image_pad = pad_image(resized_rectangle_image, scene_processing_options.out_size, scene_processing_options.out_size)
                final_mask_image, _ = pad_image(resized_rectangle_mask_image, scene_processing_options.out_size, scene_processing_options.out_size)
                data.cropped_scaled.images.append(final_image)
                data.cropped_scaled.mask_images.append(final_mask_image)
                data.cropped_scaled.pads.append(final_image_pad)
            if scene_processing_options.preserve_crops:
                final_image, final_image_pad = pad_image(cropped_image, scene_max_height, scene_max_width, mode='zero')
                final_mask_image, _ = pad_image(cropped_mask_image, scene_max_height, scene_max_width, mode='zero')
                data.cropped_unscaled.images.append(final_image)
                data.cropped_unscaled.mask_images.append(final_mask_image)
                data.cropped_unscaled.pads.append(final_image_pad)

def process_scene(scene: Scene, output_dir: Path, io_executor,
                  video_quality_evaluator: VideoQualityEvaluator, scene_processing_options: SceneProcessingOptions):
    print("processing scene", scene.id)
    cropped_scene = CroppedScene(scene, target_size=(scene_processing_options.out_size,scene_processing_options.out_size), border_size=0.08)

    data = SceneProcessingDataContainer(
        MosaicRandomParams(scene) if scene_processing_options.save_mosaic else None,
        MosaicRandomDegradationParams() if scene_processing_options.degrade_mosaic else None,
        SceneProcessingData(),
        SceneProcessingData(),
        SceneProcessingData()
    )

    scene_base_mosaic_block_size = get_base_mosaic_block_size(scene)

    scene_max_width, scene_max_height = cropped_scene.get_max_width_height()

    if scene_processing_options.save_uncropped:
        data.uncropped.images = scene.get_images()
        data.uncropped.mask_images = scene.get_masks()

    crop_processing_workers = 4
    with ThreadPoolExecutor(max_workers=crop_processing_workers) as crop_processing_executor:
        chunk_indices = list(np.linspace(0, len(cropped_scene), num=crop_processing_workers, dtype=int, endpoint=False))
        chunks = []
        crop_processing_futures = []
        for j, chunk_idx_start in enumerate(chunk_indices):
            chunk_idx_exclusive_end = chunk_indices[j+1] if chunk_idx_start != chunk_indices[-1] else len(cropped_scene)
            chunks.append(SceneProcessingDataContainer(
                data.mosaic_params,
                data.mosaic_degradation_params,
                SceneProcessingData(),
                SceneProcessingData(),
                SceneProcessingData()
            ))
            crop_processing_futures.append(crop_processing_executor.submit(process_cropped_scene, cropped_scene, scene, scene_processing_options, chunks[j], scene_max_height, scene_max_width, chunk_idx_start, chunk_idx_exclusive_end))
        wait(crop_processing_futures, return_when=ALL_COMPLETED)
        for job in futures.as_completed(crop_processing_futures):
            exception = job.exception()
            if exception:
                raise exception
        for chunk_data in chunks:
            data.cropped_scaled.images.extend(chunk_data.cropped_scaled.images)
            data.cropped_scaled.mask_images.extend(chunk_data.cropped_scaled.mask_images)
            data.cropped_scaled.pads.extend(chunk_data.cropped_scaled.pads)
            data.cropped_scaled.mosaic_images.extend(chunk_data.cropped_scaled.mosaic_images)
            data.cropped_scaled.mosaic_mask_images.extend(chunk_data.cropped_scaled.mosaic_mask_images)
            data.cropped_scaled.mosaic_pads.extend(chunk_data.cropped_scaled.mosaic_pads)

            data.cropped_unscaled.images.extend(chunk_data.cropped_unscaled.images)
            data.cropped_unscaled.mask_images.extend(chunk_data.cropped_unscaled.mask_images)
            data.cropped_unscaled.pads.extend(chunk_data.cropped_unscaled.pads)
            data.cropped_unscaled.mosaic_images.extend(chunk_data.cropped_unscaled.mosaic_images)
            data.cropped_unscaled.mosaic_mask_images.extend(chunk_data.cropped_unscaled.mosaic_mask_images)
            data.cropped_unscaled.mosaic_pads.extend(chunk_data.cropped_unscaled.mosaic_pads)

            data.uncropped.mosaic_images.extend(chunk_data.uncropped.mosaic_images)
            data.uncropped.mosaic_mask_images.extend(chunk_data.uncropped.mosaic_mask_images)

    assert_msg = "number of images in processed scene outputs should all be the same"
    if scene_processing_options.save_uncropped:
        assert len(scene) == len(data.uncropped.images) == len(data.uncropped.mask_images), assert_msg
        if scene_processing_options.save_mosaic:
            assert len(scene) == len(data.uncropped.mosaic_images) == len(data.uncropped.mosaic_mask_images), assert_msg
    if scene_processing_options.save_cropped:
        if scene_processing_options.resize_crops:
            assert len(scene) == len(data.cropped_scaled.images) == len(data.cropped_scaled.mask_images) == len(data.cropped_scaled.pads), assert_msg
            if scene_processing_options.save_mosaic:
                assert len(scene) == len(data.cropped_scaled.mosaic_images) == len(data.cropped_scaled.mosaic_mask_images) == len(data.cropped_scaled.mosaic_pads), assert_msg
        if scene_processing_options.preserve_crops:
            assert len(scene) == len(data.cropped_unscaled.images) == len(data.cropped_unscaled.mask_images) == len(data.cropped_unscaled.pads), assert_msg
            if scene_processing_options.save_mosaic:
                assert len(scene) == len(data.cropped_unscaled.mosaic_images) == len(data.cropped_unscaled.mosaic_mask_images) == len(data.cropped_unscaled.mosaic_pads), assert_msg

    #########
    ## META
    #########
    target_fps = int(round(scene.video_meta_data.video_fps))

    if scene_processing_options.save_uncropped:
        data.uncropped.meta["width"] = scene.video_meta_data.video_width
        data.uncropped.meta["height"] = scene.video_meta_data.video_height

    common_meta = []
    if scene_processing_options.save_cropped:
        if scene_processing_options.resize_crops:
            common_meta.append(data.cropped_scaled.meta)
        if scene_processing_options.preserve_crops:
            common_meta.append(data.cropped_unscaled.meta)
    if scene_processing_options.save_uncropped:
        common_meta.append(data.uncropped.meta)
    for meta in common_meta:
        meta["fps"] = target_fps
        meta["frames_count"] = len(scene)
        meta["name"] = scene.file_path.name
        meta["orig_width"] = scene.video_meta_data.video_width
        meta["orig_height"] = scene.video_meta_data.video_height
        meta["base_mosaic_block_size"] = scene_base_mosaic_block_size

    if scene_processing_options.save_mosaic:
        for meta in common_meta:
            meta["mosaic"] = {}
            meta["mosaic"]["mod"] = data.mosaic_params.mosaic_mod
            meta["mosaic"]["rect_ratio"] = data.mosaic_params.mosaic_rectangle_ratio
            meta["mosaic"]["mosaic_size"] = data.mosaic_params.mosaic_size
            meta["mosaic"]["feather_size"] = data.mosaic_params.mosaic_feather_size

    if scene_processing_options.save_cropped:
        if scene_processing_options.resize_crops:
            data.cropped_scaled.quality_score = video_quality_evaluator.evaluate(data.cropped_scaled.images)
        if scene_processing_options.preserve_crops:
            data.cropped_unscaled.quality_score = video_quality_evaluator.evaluate(data.cropped_unscaled.images)
    if scene_processing_options.save_uncropped:
        data.uncropped.quality_score = video_quality_evaluator.evaluate(data.uncropped.images)

    if scene_processing_options.save_cropped:
        if scene_processing_options.resize_crops:
            data.cropped_scaled.meta["pad"] = data.cropped_scaled.pads
            data.cropped_scaled.meta["height"], data.cropped_scaled.meta["width"] = data.cropped_scaled.images[0].shape[:2]
            data.cropped_scaled.meta["video_quality"] = data.cropped_scaled.quality_score
        if scene_processing_options.preserve_crops:
            data.cropped_unscaled.meta["pad"] = data.cropped_unscaled.pads
            data.cropped_unscaled.meta["height"], data.cropped_unscaled.meta["width"] = data.cropped_unscaled.images[0].shape[:2]
            data.cropped_unscaled.meta["video_quality"] = data.cropped_unscaled.quality_score
    if scene_processing_options.save_uncropped:
        data.uncropped.meta["video_quality"] = data.uncropped.quality_score

    #########
    ## IO
    #########
    file_suffix = '-'
    io_futures = []
    def _save_imgs(imgs, name, file_extension):
        frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, name, scene.file_path.name, scene.id,
                                                         scene_processing_options.save_flat,
                                                         file_suffix, video=False)
        io_futures.append(io_executor.submit(save_imgs, frame_dir, file_prefix, file_extension, imgs))
    def _save_vid(imgs, name, gray=False):
        frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, name, scene.file_path.name, scene.id,
                                                         scene_processing_options.save_flat,
                                                         file_suffix, video=True)
        io_futures.append(io_executor.submit(save_vid, frame_dir, file_prefix, imgs, target_fps, gray))
    def _save_meta(meta, name):
        frame_dir, file_prefix = get_dir_and_file_prefix(output_dir, name, scene.file_path.name, scene.id,
                                                         scene_processing_options.save_flat,
                                                         file_suffix, video=True)
        io_futures.append(io_executor.submit(save_meta, frame_dir, file_prefix, meta))

    if scene_processing_options.save_cropped:
        if scene_processing_options.resize_crops:
            _save_meta(data.cropped_scaled.meta, "crop_scaled_meta")
            _save_imgs(data.cropped_scaled.images, "crop_scaled_img", ".jpg") if scene_processing_options.save_as_images else _save_vid(data.cropped_scaled.images, "crop_scaled_img")
            _save_imgs(data.cropped_scaled.mask_images, "crop_scaled_mask", ".png") if scene_processing_options.save_as_images else _save_vid(data.cropped_scaled.mask_images, "crop_scaled_mask", gray=True)
            if scene_processing_options.save_mosaic:
                _save_imgs(data.cropped_scaled.mosaic_images, "crop_scaled_mosaic", ".jpg") if scene_processing_options.save_as_images else _save_vid(data.cropped_scaled.mosaic_images, "crop_scaled_mosaic")
                _save_imgs(data.cropped_scaled.mosaic_mask_images, "crop_scaled_mask_mosaic", ".png") if scene_processing_options.save_as_images else _save_vid(data.cropped_scaled.mosaic_mask_images, "crop_scaled_mask_mosaic", gray=True)
        if scene_processing_options.preserve_crops:
            _save_meta(data.cropped_unscaled.meta, "crop_unscaled_meta")
            _save_imgs(data.cropped_unscaled.images, "crop_unscaled_img", ".jpg") if scene_processing_options.save_as_images else _save_vid(data.cropped_unscaled.images, "crop_unscaled_img")
            _save_imgs(data.cropped_unscaled.mask_images, "crop_unscaled_mask", ".png") if scene_processing_options.save_as_images else _save_vid(data.cropped_unscaled.mask_images, "crop_unscaled_mask", gray=True)
            if scene_processing_options.save_mosaic:
                _save_imgs(data.cropped_unscaled.mosaic_images, "crop_unscaled_mosaic", ".jpg") if scene_processing_options.save_as_images else _save_vid(data.cropped_unscaled.mosaic_images, "crop_unscaled_mosaic")
                _save_imgs(data.cropped_unscaled.mosaic_mask_images, "crop_unscaled_mask_mosaic", ".png") if scene_processing_options.save_as_images else _save_vid(data.cropped_unscaled.mosaic_mask_images, "crop_unscaled_mask_mosaic", gray=True)
    if scene_processing_options.save_uncropped:
        _save_meta(data.uncropped.meta, "orig_meta")
        _save_imgs(data.uncropped.images, "orig_img", ".jpg") if scene_processing_options.save_as_images else _save_vid(data.uncropped.images, "orig_img")
        _save_imgs(data.uncropped.mask_images, "orig_mask", ".png") if scene_processing_options.save_as_images else _save_vid(data.uncropped.mask_images, "orig_mask", gray=True)
        if scene_processing_options.save_mosaic:
            _save_imgs(data.uncropped.mosaic_images, "orig_mosaic", ".jpg") if scene_processing_options.save_as_images else _save_vid(data.uncropped.mosaic_images, "orig_mosaic")
            _save_imgs(data.uncropped.mosaic_mask_images, "orig_mask_mosaic", ".png") if scene_processing_options.save_as_images else _save_vid(data.uncropped.mosaic_mask_images, "orig_mask_mosaic", gray=True)

    wait(io_futures, return_when=ALL_COMPLETED)
    print("done processing scene", scene.id)


def process_file(model: ultralytics.models.yolo.model.Model, video_metadata: VideoMetadata, output_dir: Path,
                 scenes_executor, io_executor, video_quality_evaluator,
                 file_processing_options: FileProcessingOptions,
                 scene_processing_options: SceneProcessingOptions,
                 scene_executor_worker_count: int,
                 model_device=None):
    scene_futures = []
    for scene in SceneGenerator(NsfwFramesGenerator(model, video_metadata, model_device, stride_mode_activation_length=None, stride_length=None),
                                file_processing_options.scene_min_length, file_processing_options.scene_max_length,
                                random_extend_masks=True, stride_length=file_processing_options.stride_length)():
        print(
            f"found scene {scene.id} (frames {scene.frame_start:06d}-{scene.frame_end:06d}), queuing up for processing")
        scene_futures.append(
            scenes_executor.submit(process_scene, scene, output_dir,
                                   io_executor, video_quality_evaluator, scene_processing_options))
        while len([future for future in scene_futures if not future.done()]) >= scene_executor_worker_count + 1:
            # print(f"workers busy, block until they are available: running {len([future for future in scene_futures if future.running()])}, lets get to work: {len([future for future in scene_futures if not future.done()])}")
            sleep(1)
            pass  # do nothing until workers become available. Otherwise, we could queue up a lot of futures which use a lot of memory as we pass Scene objects
        # we don't care about done futures, lets clean them up to potentially free memory
        clean_up_completed_futures(scene_futures)
        # print(f"deleted done future. futures now {len(scene_futures)}")
    return scene_futures

def clean_up_completed_futures(completed_futures):
    for job in futures.as_completed(completed_futures):
        exception = job.exception()
        if exception:
            print(f"ERR(main): failed processing scene: {type(exception).__name__}: {exception}")
            raise exception
        completed_futures.remove(job)

def determine_max_scene_length(video_metadata: VideoMetadata, limit_seconds: int | None, limit_memory: int | None):
    scene_max_length = None
    if limit_seconds:
        scene_max_length = limit_seconds
    if limit_memory:
        scene_max_length_memory = video_utils.approx_max_length_by_memory_limit(video_metadata, limit_memory)
        scene_max_length = min(scene_max_length, scene_max_length_memory) if scene_max_length else scene_max_length_memory
    return scene_max_length

def parse_args():
    parser = argparse.ArgumentParser("Create video dataset")
    parser.add_argument('--output-root', type=Path, default='video_dataset', help="set output root directory")
    parser.add_argument('--input', type=Path, help="path to video file or directory")
    parser.add_argument('--model', type=str, default="model_weights/lada_nsfw_detection_model.pt",
                        help="path to YOLO model")
    parser.add_argument('--model-device', type=str, default="cuda", help="device to run the YOLO model on. E.g. 'cuda' or 'cuda:0'")
    parser.add_argument('--quality-model-device', type=str, default="cuda", help="device to run the video quality model on. E.g. 'cuda' or 'cuda:0'")
    parser.add_argument('--workers', type=int, default=4, help="Set number of multiprocessing workers")
    parser.add_argument('--start-index', type=int, default=0, help="Can be used to continue a previous run. Note the index number next to last processed file name")
    parser.add_argument('--scene-min-length', type=int, default=2.,
                        help="minimal length of a scene in number of frames in order to be detected (in seconds)")
    parser.add_argument('--scene-max-length', type=int, default=8,
                        help="maximum length of a scene in number of frames. Scenes longer than that will be cut (in seconds)")
    parser.add_argument('--scene-max-memory', default=6144, type=int, help="limits maximum scene length based on approximate memory consumption of the scene. Value should be given in Megabytes (MB)")
    parser.add_argument('--out-size', type=int, default=256, help="size in pixel of output images")
    parser.add_argument('--flat', default=True, action=argparse.BooleanOptionalAction,
                        help="Store frames of all videos in output root directory instead of using sub directories per clip")
    parser.add_argument('--save-uncropped', default=False, action=argparse.BooleanOptionalAction,
                        help="Save uncropped, full-size images and masks")
    parser.add_argument('--save-cropped', default=True, action=argparse.BooleanOptionalAction,
                        help="Save cropped images and masks")
    parser.add_argument('--stride-mode-activation-length', default=0, type=int, help="start activating stride-mode instead of processing every frame in the video when video is at least this long. values is in seconds")
    parser.add_argument('--stride-length', default=0, type=int, help="skip frames in between long videos to prevent sampling too many scene from a single file. value is in seconds")
    parser.add_argument('--save-mosaic', default=False, action=argparse.BooleanOptionalAction,
                        help="Create and save mosaic images and masks")
    parser.add_argument('--resize-crops', default=True, action=argparse.BooleanOptionalAction,
                        help="Resize crops to out-size. adds padding if necessary")
    parser.add_argument('--preserve-crops', default=True, action=argparse.BooleanOptionalAction,
                        help="Keeps scale/resolution of cropped scenes. adds padding if necessary")
    parser.add_argument('--save-as-images', default=False, action=argparse.BooleanOptionalAction,
                        help="Save as images instead of videos")
    parser.add_argument('--degrade-mosaic', default=False, action=argparse.BooleanOptionalAction,
                        help="degrades mosaic and clean video clips to better match real world samples")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if not (args.save_cropped or args.save_uncropped):
        print("No save option given. Specify at least one!")
        return

    io_executor = ThreadPoolExecutor(max_workers=4)
    scenes_executor = ThreadPoolExecutor(max_workers=args.workers)

    yolo_device = args.model_device
    video_quality_evaluator_device = args.quality_model_device

    model = YOLO(args.model)
    video_quality_evaluator = VideoQualityEvaluator(device=video_quality_evaluator_device)

    output_dir = args.output_root
    if not output_dir.exists():
        output_dir.mkdir()
    input_path = args.input
    video_files = input_path.glob("*") if input_path.is_dir() else [input_path]
    scene_processing_futures = []

    for file_index, file_path in enumerate(video_files):
        if file_index < args.start_index or len(list(output_dir.glob(f"*/{file_path.name}*"))) > 0:
            print(f"{file_index}, Skipping {file_path.name}: Already processed")
            continue
        if not video_utils.is_video_file(file_path):
            print(f"{file_index}, Skipping {file_path.name}: Unsupported file format")
            continue
        video_metadata = video_utils.get_video_meta_data(file_path)
        scene_max_length = determine_max_scene_length (video_metadata, args.scene_max_length, args.scene_max_memory)
        if scene_max_length < args.scene_min_length:
            print(f"{file_index}, Skipping {file_path.name}: Scene maximum length is less than minimum length")
            continue

        print(f"{file_index}, Processing {file_path.name}")

        scene_processing_options = SceneProcessingOptions(output_dir=output_dir,
                                                          save_flat=args.flat,
                                                          out_size=args.out_size,
                                                          save_cropped=args.save_cropped,
                                                          save_uncropped=args.save_uncropped,
                                                          resize_crops=args.resize_crops,
                                                          preserve_crops=args.preserve_crops,
                                                          save_mosaic=args.save_mosaic,
                                                          degrade_mosaic=args.degrade_mosaic,
                                                          save_as_images=args.save_as_images)

        file_processing_options = FileProcessingOptions(scene_max_length=scene_max_length,
                                                        scene_min_length=args.scene_min_length,
                                                        stride_length=args.stride_length)

        scene_processing_futures.extend(process_file(model, video_metadata, output_dir, scenes_executor,
                     io_executor, video_quality_evaluator, file_processing_options, scene_processing_options, args.workers, yolo_device))
        clean_up_completed_futures(scene_processing_futures)

    wait(scene_processing_futures, return_when=ALL_COMPLETED)

    io_executor.shutdown(wait=True)
    scenes_executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
