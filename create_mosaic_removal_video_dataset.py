import argparse
import pathlib
from concurrent import futures
from concurrent.futures import wait, ALL_COMPLETED, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Literal, Optional

import cv2
import numpy as np
import ultralytics.engine.results
from ultralytics import YOLO

from lada.lib import VideoMetadata, mask_utils, restoration_dataset_metadata, Pad, Mask, Image
from lada.lib import video_utils, image_utils
from lada.lib.degradation_utils import MosaicRandomDegradationParams, apply_frame_degradation
from lada.dover.evaluate import VideoQualityEvaluator
from lada.lib.image_utils import pad_image
from lada.lib.mosaic_utils import get_random_parameter, addmosaic_base, get_mosaic_block_size_v1, \
    get_mosaic_block_size_v2, get_mosaic_block_size_v3
from lada.lib.nsfw_scene_detector import NsfwSceneGenerator, Scene, CroppedScene
from lada.lib.nudenet_nsfw_detector import NudeNetNsfwDetector
from lada.lib.ultralytics_utils import disable_ultralytics_telemetry
from lada.lib.watermark_detector import WatermarkDetector

disable_ultralytics_telemetry()

@dataclass
class VideoQualityProcessingOptions:
    filter: bool
    add_metadata: bool
    min_quality: float

@dataclass
class WatermarkDetectionProcessingOptions:
    filter: bool
    add_metadata: bool

@dataclass
class NudeNetNsfwDetectionProcessingOptions:
    filter: bool
    add_metadata: bool

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
    quality_evaluation: VideoQualityProcessingOptions
    watermark_detection: WatermarkDetectionProcessingOptions
    nudenet_nsfw_detection: NudeNetNsfwDetectionProcessingOptions

@dataclass
class FileProcessingOptions:
    scene_min_length: int
    scene_max_length: int
    stride_length: int


def get_base_mosaic_block_size(scene: Scene) -> restoration_dataset_metadata.MosaicBlockSizeV2:
    box_sizes = [(r - l + 1) * (b - t + 1) for t, l, b, r in scene.get_boxes()]
    median_idx = np.argsort(box_sizes)[len(box_sizes) // 2]
    _, mask_image_representative, box = scene[median_idx]
    t, l, b, r = box
    filter = np.ones_like(mask_image_representative, dtype=bool)
    filter[t:b + 1, l:r + 1, :] = False
    mask_image_representative[filter] = 0

    # not sure what we'll use later, lets save all variants for now

    mosaic_block_size = restoration_dataset_metadata.MosaicBlockSizeV2(
        mosaic_size_v3=get_mosaic_block_size_v3((scene.video_meta_data.video_height, scene.video_meta_data.video_width)),
        mosaic_size_v2=get_mosaic_block_size_v2(mask_image_representative),
        mosaic_size_v1_normal=get_mosaic_block_size_v1(mask_image_representative, 'normal'),
        mosaic_size_v1_bounding=get_mosaic_block_size_v1(mask_image_representative, 'bounding'))
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

class DatasetItem:
    def __init__(self, cropped_scene: CroppedScene, scene: Optional[Scene], mosaic: bool, crop: bool, resize_crops: Optional[bool], resize_crop_size: Optional[int], mosaic_params: Optional[MosaicRandomParams], mosaic_degradation_params: Optional[MosaicRandomDegradationParams], scene_type: Literal['cropped_scaled', 'cropped_unscaled', 'uncropped']):
        self._images: list[Image] = []
        self._masks: list[Mask] = []
        self._pads: list[Pad] = []
        self._meta: Optional[restoration_dataset_metadata.RestorationDatasetMetadataV2] = None
        self._quality_score: Optional[restoration_dataset_metadata.VisualQualityScoreV1] = None
        self._watermark_detected: Optional[bool] = None
        self._nudenet_nsfw_detected: Optional[bool] = None
        self._nudenet_nsfw_detected_classes: Optional[restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1] = None
        self._scene_type: Literal['cropped_scaled', 'cropped_unscaled', 'uncropped'] = scene_type

        self._init_images_masks_and_pad(cropped_scene, scene, mosaic, crop, resize_crops, resize_crop_size, mosaic_params, mosaic_degradation_params)

    @property
    def images(self):
        return self._images

    @property
    def masks(self):
        return self._masks

    @property
    def pads(self):
        return self._pads

    @property
    def quality_score(self):
        return self._quality_score

    @quality_score.setter
    def quality_score(self, value):
        self._quality_score = value

    @property
    def watermark_detected(self):
        return self._watermark_detected

    @watermark_detected.setter
    def watermark_detected(self, value):
        self._watermark_detected = value

    @property
    def nudenet_nsfw_detected(self):
        return self._nudenet_nsfw_detected

    @nudenet_nsfw_detected.setter
    def nudenet_nsfw_detected(self, value):
        self._nudenet_nsfw_detected = value

    @property
    def nudenet_nsfw_detected_classes(self):
        return self._nudenet_nsfw_detected_classes

    @nudenet_nsfw_detected_classes.setter
    def nudenet_nsfw_detected_classes(self, value):
        self._nudenet_nsfw_detected_classes = value

    def _init_images_masks_and_pad(self, cropped_scene: CroppedScene, scene: Optional[Scene], mosaic: bool, crop: bool, resize_crops: bool, resize_crop_size: Optional[int], mosaic_params: Optional[MosaicRandomParams], mosaic_degradation_params: Optional[MosaicRandomDegradationParams]):

        for i in range(len(cropped_scene)):
            if mosaic:
                scene_image, scene_mask, _ = cropped_scene[i]
                mask = mask_utils.dilate_mask(scene_mask, iterations=mosaic_params.mosaic_mask_dilation_iterations)
                cropped_image, cropped_mask = addmosaic_base(scene_image, mask, mosaic_params.mosaic_size,
                                                                           model=mosaic_params.mosaic_mod, rect_ratio=mosaic_params.mosaic_rectangle_ratio,
                                                                           feather=mosaic_params.mosaic_feather_size)
            else:
                cropped_image, cropped_mask, _ = cropped_scene[i]

            if crop:
                image, mask = cropped_image, cropped_mask
                if mosaic and mosaic_degradation_params:
                    image = apply_frame_degradation(image, mosaic_degradation_params)
                if resize_crops:
                    mask = image_utils.resize(mask, resize_crop_size, interpolation=cv2.INTER_NEAREST)
                    image = image_utils.resize(image, resize_crop_size, interpolation=cv2.INTER_CUBIC)
                    mask, _ = pad_image(mask, resize_crop_size, resize_crop_size, mode='zero')
                    image, pad = pad_image(image, resize_crop_size, resize_crop_size, mode='zero')
                else:
                    max_width, max_height = cropped_scene.get_max_width_height()
                    mask, _ = pad_image(mask, max_height, max_width, mode='zero')
                    image, pad = pad_image(image, max_height, max_width, mode='zero')
            else:
                scene_image, scene_mask, scene_box = scene[i]
                if mosaic:
                    _, _, scene_box = cropped_scene[i]
                    t, l, b, r = scene_box
                    image = scene_image.copy()
                    image[t:b + 1, l:r + 1, :] = cropped_image
                    if mosaic_degradation_params:
                        image = apply_frame_degradation(image, mosaic_degradation_params)
                    mask = np.zeros_like(scene_mask, dtype=scene_mask.dtype)
                    mask[t:b + 1, l:r + 1] = cropped_mask
                else:
                    image, mask = scene_image, scene_mask
                pad = [0, 0, 0, 0]

            self._images.append(image)
            self._masks.append(mask)
            self._pads.append(pad)

        assert len(cropped_scene) == len(self._images) == len(self._masks) == len(self._pads), f"number of images, masks and pads are not the same: {len(cropped_scene)} == {len(self._images)} == {len(self._masks)} == {len(self._pads)}"

    def init_meta(self, scene: Scene, scene_base_mosaic_block_size: restoration_dataset_metadata.MosaicBlockSizeV2, output_dir, save_as_images: bool, save_flat: bool, mosaic: bool, mosaic_params: MosaicRandomParams):
        def _get_relative_path_dir(scene_type, mosaic):
            metadata_path = get_io_path(output_dir, scene_type, scene, save_as_images, save_flat, 'meta', mosaic)
            img_path = get_io_path(output_dir, scene_type, scene, save_as_images, save_flat, 'img', mosaic)
            mask_path = get_io_path(output_dir, scene_type, scene, save_as_images, save_flat, 'mask', mosaic)
            return str(img_path.relative_to(metadata_path.parent, walk_up=True)), str(mask_path.relative_to(metadata_path.parent, walk_up=True))

        mosaic_metadata = restoration_dataset_metadata.MosaicMetadataV1(mod=mosaic_params.mosaic_mod,
                                                                        rect_ratio=mosaic_params.mosaic_rectangle_ratio,
                                                                        mosaic_size=mosaic_params.mosaic_size,
                                                                        feather_size=mosaic_params.mosaic_feather_size) if mosaic else None

        relative_nsfw_video_path, relative_mask_video_path = _get_relative_path_dir(self._scene_type, False)
        relative_mosaic_nsfw_video_path, relative_mosaic_mask_video_path = _get_relative_path_dir(self._scene_type,True) if mosaic else (None, None)
        self._meta = restoration_dataset_metadata.RestorationDatasetMetadataV2(
            name=scene.file_path.name,
            fps=scene.video_meta_data.video_fps,
            frames_count=len(scene),
            orig_shape=(scene.video_meta_data.video_height, scene.video_meta_data.video_width),
            scene_shape=self._images[0].shape[:2],
            base_mosaic_block_size=scene_base_mosaic_block_size,
            pad=self._pads,
            relative_nsfw_video_path=relative_nsfw_video_path,
            relative_mask_video_path=relative_mask_video_path,
            relative_mosaic_nsfw_video_path=relative_mosaic_nsfw_video_path,
            relative_mosaic_mask_video_path=relative_mosaic_mask_video_path,
            mosaic=mosaic_metadata,
            video_quality=self._quality_score,
            watermark_detected=self._watermark_detected,
            nudenet_nsfw_detected=self._nudenet_nsfw_detected,
            nudenet_nsfw_detected_classes=self._nudenet_nsfw_detected_classes,
        )

    def save(self, output_dir, scene, mosaic, save_as_images, save_flat, target_fps, io_executor):
        io_futures = []

        def _save(data, file_type: Literal['mask', 'img', 'meta'], mosaic):
            file_path = get_io_path(output_dir, self._scene_type, scene, save_as_images, save_flat, file_type, mosaic)
            if file_type == 'meta':
                io_futures.append(io_executor.submit(save_meta, file_path, data))
            else:
                if save_as_images:
                    io_futures.append(io_executor.submit(save_imgs, file_path, data))
                else:
                    gray = file_type == 'mask'
                    io_futures.append(io_executor.submit(save_vid, file_path, data, target_fps, gray))
        if self._meta:
            _save(self._meta, 'meta', mosaic)
        _save(self._images, 'img', mosaic)
        _save(self._masks, 'mask', mosaic)

        wait(io_futures, return_when=ALL_COMPLETED)

def save_vid(file_path: pathlib.Path, imgs: list[np.ndarray[np.uint8]], fps=30, gray=False):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    path = str(file_path.absolute())
    if gray:
        video_utils.write_masks_to_video_file(imgs, path, fps)
    else:
        video_utils.write_frames_to_video_file(imgs, path, fps)

def save_meta(file_path: pathlib.Path, meta: restoration_dataset_metadata.RestorationDatasetMetadataV2):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    path = str(file_path.absolute())
    meta.to_json_file(path)

def save_imgs(file_path_template_format_string: pathlib.Path, imgs: np.ndarray, jpeg_quality_level=95):
    file_path_template_format_string.parent.mkdir(parents=True, exist_ok=True)
    file_ext = file_path_template_format_string
    for i in range(len(imgs)):
        path = str(Path(str(file_path_template_format_string).format(i)).absolute())
        try:
            if file_ext == ".jpg":
                cv2.imwrite(path, imgs[i], [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_level])
            else:
                cv2.imwrite(path, imgs[i])
                cv2.imwrite(path, imgs[i])
        except Exception as e:
            print(e)

def get_io_path(output_dir:pathlib.Path, scene_type: Literal['cropped_scaled', 'cropped_unscaled', 'uncropped'], scene:Scene, save_as_images:bool, save_flat:bool, file_type: Literal['mask', 'img', 'meta'], mosaic: bool) -> pathlib.Path:
    file_suffix = '-'
    subdir_name = 'crop_scaled' if scene_type == 'cropped_scaled' else 'crop_unscaled' if scene_type == 'cropped_unscaled' else 'orig'
    if not (mosaic and file_type == 'img'):
        subdir_name += ('_' + file_type)
    if mosaic:
        subdir_name += "_mosaic"
    file_name = scene.file_path.name
    if save_flat:
        file_prefix = f"{file_name}-{scene.id:06d}{file_suffix}"
        frame_dir = output_dir.joinpath(subdir_name)
    else:
        if save_as_images:
            file_prefix = ""
            frame_dir = output_dir.joinpath(subdir_name).joinpath(file_name).joinpath(f"{scene.id:06d}")
        else:
            file_prefix = f"{scene.id:06d}"
            frame_dir = output_dir.joinpath(subdir_name).joinpath(file_name)
    if file_type == 'img':
        if save_as_images:
            file_extension = ".jpg"
            file_name_format_string = file_prefix + '{%06d}' + file_extension
            file_name = file_name_format_string
        else:
            file_extension = ".mp4"
            file_name = file_prefix + file_extension
    elif file_type == 'mask':
        if save_as_images:
            file_extension = ".png"
            file_name_format_string = file_prefix + '{%06d}' + file_extension
            file_name = file_name_format_string
        else:
            file_extension = ".mkv"
            file_name = file_prefix + file_extension
    elif file_type == 'meta':
        file_extension = ".json"
        file_name = file_prefix + file_extension
    else:
        raise TypeError("expected file_type to be one of: 'mask', 'img' or 'meta'")
    return frame_dir.joinpath(file_name)

def process_scene(scene: Scene, output_dir: Path, io_executor,
                  video_quality_evaluator: VideoQualityEvaluator, watermark_detector: WatermarkDetector, nudenet_nsfw_detector: NudeNetNsfwDetector, scene_processing_options: SceneProcessingOptions):
    print("Started processing scene", scene.id)
    cropped_scene = CroppedScene(scene, target_size=(scene_processing_options.out_size,scene_processing_options.out_size), border_size=0.08)

    dataset_item_mosaic_crop_unscaled: Optional[DatasetItem] = None
    dataset_item_mosaic_crop_scaled: Optional[DatasetItem] = None
    dataset_item_mosaic_uncropped: Optional[DatasetItem] = None
    dataset_item_crop_unscaled: Optional[DatasetItem] = None
    dataset_item_crop_scaled: Optional[DatasetItem] = None
    dataset_item_uncropped: Optional[DatasetItem] = None

    mosaic_params = MosaicRandomParams(scene) if scene_processing_options.save_mosaic else None

    #########
    ## Images, Masks, Pads
    #########
    if scene_processing_options.save_mosaic:
        mosaic_params = MosaicRandomParams(scene)
        degradation_params = MosaicRandomDegradationParams() if scene_processing_options.degrade_mosaic else None
        if scene_processing_options.save_cropped:
            if scene_processing_options.preserve_crops:
                dataset_item_mosaic_crop_unscaled = DatasetItem(cropped_scene, scene, True, True, False, scene_processing_options.out_size, mosaic_params, degradation_params, 'cropped_unscaled')
            if scene_processing_options.resize_crops:
                dataset_item_mosaic_crop_scaled = DatasetItem(cropped_scene, scene, True, True, True, scene_processing_options.out_size, mosaic_params, degradation_params, 'cropped_scaled')
        if scene_processing_options.save_uncropped:
            dataset_item_mosaic_uncropped = DatasetItem(cropped_scene, scene, True, False, None, None, mosaic_params, degradation_params, 'uncropped')
    if scene_processing_options.save_cropped:
        if scene_processing_options.save_cropped:
            if scene_processing_options.preserve_crops:
                dataset_item_crop_unscaled = DatasetItem(cropped_scene, scene, False, True, False, scene_processing_options.out_size, None, None,'cropped_unscaled')
            if scene_processing_options.resize_crops:
                dataset_item_crop_scaled = DatasetItem(cropped_scene, scene, False, True, True, scene_processing_options.out_size, None, None, 'cropped_scaled')
        if scene_processing_options.save_uncropped:
            dataset_item_uncropped = DatasetItem(cropped_scene, scene, False, False, None, None, None, None, 'uncropped')


    #########
    ## Video quality evaluation
    #########
    if scene_processing_options.quality_evaluation.filter or scene_processing_options.quality_evaluation.add_metadata:
        if scene_processing_options.save_cropped:
            if scene_processing_options.resize_crops:
                score = video_quality_evaluator.evaluate(dataset_item_crop_scaled.images)
                dataset_item_crop_scaled.quality_score = restoration_dataset_metadata.VisualQualityScoreV1(**score)
            if scene_processing_options.preserve_crops:
                score = video_quality_evaluator.evaluate(dataset_item_crop_unscaled.images)
                dataset_item_crop_unscaled.quality_score = restoration_dataset_metadata.VisualQualityScoreV1(**score)
        if scene_processing_options.save_uncropped:
            score = video_quality_evaluator.evaluate(dataset_item_uncropped.images)
            dataset_item_uncropped.quality_score = restoration_dataset_metadata.VisualQualityScoreV1(**score)

    #########
    ## Watermark detection
    #########
    if scene_processing_options.watermark_detection.filter or scene_processing_options.watermark_detection.add_metadata:
        if scene_processing_options.save_cropped:
            _watermark_detected = watermark_detector.detect(scene.get_images(), cropped_scene.get_boxes())
            if scene_processing_options.resize_crops:
                dataset_item_crop_scaled.watermark_detected = _watermark_detected
            if scene_processing_options.preserve_crops:
                dataset_item_crop_unscaled.watermark_detected = _watermark_detected
        if scene_processing_options.save_uncropped:
            dataset_item_uncropped.watermark_detected = watermark_detector.detect(scene.get_images())

    #########
    ## NudeNet NSFW detection
    #########
    if scene_processing_options.nudenet_nsfw_detection.filter or scene_processing_options.nudenet_nsfw_detection.add_metadata:
        if scene_processing_options.save_cropped:
            _nudenet_nsfw_detected, _nsfw_male_detected, _nsfw_female_detected = nudenet_nsfw_detector.detect(scene.get_images(), cropped_scene.get_boxes())
            if scene_processing_options.resize_crops:
                dataset_item_crop_scaled.nudenet_nsfw_detected, dataset_item_crop_scaled.nudenet_nsfw_detected_classes = _nudenet_nsfw_detected, restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1(_nsfw_male_detected, _nsfw_female_detected)
            if scene_processing_options.preserve_crops:
                dataset_item_crop_unscaled.nudenet_nsfw_detected, dataset_item_crop_unscaled.nudenet_nsfw_detected_classes = _nudenet_nsfw_detected, restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1(_nsfw_male_detected, _nsfw_female_detected)
        if scene_processing_options.save_uncropped:
            _nudenet_nsfw_detected, _nsfw_male_detected, _nsfw_female_detected = nudenet_nsfw_detector.detect(scene.get_images())
            dataset_item_uncropped.nudenet_nsfw_detected, dataset_item_crop_scaled.nudenet_nsfw_detected_classes = _nudenet_nsfw_detected, restoration_dataset_metadata.NudeNetNsfwClassDetectionsV1(_nsfw_male_detected, _nsfw_female_detected)

    #########
    ## META
    #########
    scene_base_mosaic_block_size = get_base_mosaic_block_size(scene)
    if scene_processing_options.save_cropped:
        if scene_processing_options.resize_crops:
            dataset_item_crop_scaled.init_meta(scene, scene_base_mosaic_block_size, output_dir, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene_processing_options.save_mosaic, mosaic_params)
        if scene_processing_options.preserve_crops:
            dataset_item_crop_unscaled.init_meta(scene, scene_base_mosaic_block_size, output_dir, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene_processing_options.save_mosaic, mosaic_params)
    if scene_processing_options.save_uncropped:
        dataset_item_uncropped.init_meta(scene, scene_base_mosaic_block_size, output_dir, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene_processing_options.save_mosaic, mosaic_params)

    #########
    ## Filtering
    #########
    if scene_processing_options.save_cropped and scene_processing_options.preserve_crops:
        scene_quality = dataset_item_crop_unscaled.quality_score.overall
        watermark_detected = dataset_item_crop_unscaled.watermark_detected
        nudenet_nsfw_detected = dataset_item_crop_unscaled.nudenet_nsfw_detected
    elif scene_processing_options.save_cropped and scene_processing_options.resize_crops:
        scene_quality = dataset_item_crop_scaled.quality_score.overall
        watermark_detected = dataset_item_crop_scaled.watermark_detected
        nudenet_nsfw_detected = dataset_item_crop_scaled.nudenet_nsfw_detected
    elif scene_processing_options.save_uncropped:
        scene_quality = dataset_item_uncropped.quality_score.overall
        watermark_detected = dataset_item_uncropped.watermark_detected
        nudenet_nsfw_detected = dataset_item_uncropped.nudenet_nsfw_detected
    else:
        scene_quality = 1.0
        watermark_detected = None
        nudenet_nsfw_detected = None

    if scene_processing_options.quality_evaluation.filter and scene_quality < scene_processing_options.quality_evaluation.min_quality:
        print(f"Skipped scene {scene.id} because of low visual video quality ({scene_quality:.4f} < {scene_processing_options.quality_evaluation.min_quality})")
    elif scene_processing_options.watermark_detection.filter and watermark_detected:
        print(f"Skipped scene {scene.id} because watermark(s) have been detected")
    elif scene_processing_options.nudenet_nsfw_detection.filter and not nudenet_nsfw_detected:
        print(f"Skipped scene {scene.id} because not NSFW according to NudeNetNsfwDetector")
    else:
        if scene_processing_options.save_mosaic:
            if scene_processing_options.save_cropped:
                if scene_processing_options.preserve_crops:
                    dataset_item_mosaic_crop_unscaled.save(output_dir, scene, True, scene_processing_options.save_as_images, scene_processing_options.save_flat, scene.video_meta_data.video_fps, io_executor)
                if scene_processing_options.resize_crops:
                    dataset_item_mosaic_crop_scaled.save(output_dir, scene, True, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor)
            if scene_processing_options.save_uncropped:
                dataset_item_mosaic_uncropped.save(output_dir, scene, True, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor)
        if scene_processing_options.save_cropped:
            if scene_processing_options.preserve_crops:
                dataset_item_crop_unscaled.save(output_dir, scene, False, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor)
            if scene_processing_options.resize_crops:
                dataset_item_crop_scaled.save(output_dir, scene, False, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor)
        if scene_processing_options.save_uncropped:
            dataset_item_uncropped.save(output_dir, scene, False, scene_processing_options.save_as_images, scene_processing_options.save_flat,  scene.video_meta_data.video_fps, io_executor)
        print("Finished processing scene", scene.id)

def process_file(model: ultralytics.models.yolo.model.Model, video_metadata: VideoMetadata, output_dir: Path,
                 scenes_executor, io_executor, video_quality_evaluator, watermark_detector: WatermarkDetector,
                 nudenet_nsfw_detector: NudeNetNsfwDetector,
                 file_processing_options: FileProcessingOptions,
                 scene_processing_options: SceneProcessingOptions,
                 scene_executor_worker_count: int,
                 model_device=None):
    scene_futures = []
    for scene in NsfwSceneGenerator(model, video_metadata, model_device,
                                    file_processing_options.scene_min_length, file_processing_options.scene_max_length,
                                    random_extend_masks=True, stride_length=file_processing_options.stride_length)():
        print(f"Found scene {scene.id} (frames {scene.frame_start:06d}-{scene.frame_end:06d}), queuing up for processing")
        scene_futures.append(
            scenes_executor.submit(process_scene, scene, output_dir,
                                   io_executor, video_quality_evaluator, watermark_detector, nudenet_nsfw_detector, scene_processing_options))
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
    parser = argparse.ArgumentParser("Create mosaic restoration dataset")
    parser.add_argument('--workers', type=int, default=4, help="Set number of multiprocessing workers")

    input = parser.add_argument_group('Input')
    input.add_argument('--input', type=Path, help="path to a video file or a directory containing NSFW videos")
    input.add_argument('--start-index', type=int, default=0, help="Can be used to continue a previous run. Note the index number next to last processed file name")
    input.add_argument('--stride-length', default=0, type=int, help="skip frames in between long videos to prevent sampling too many scenes from a single file. value is in seconds")

    output = parser.add_argument_group('Output')
    output.add_argument('--output-root', type=Path, default='video_dataset', help="path to directory where dataset should be stored")
    output.add_argument('--out-size', type=int, default=256, help="size (in pixel) of output images")
    output.add_argument('--save-uncropped', default=False, action=argparse.BooleanOptionalAction,
                        help="Save uncropped, full-size images and masks")
    output.add_argument('--save-cropped', default=True, action=argparse.BooleanOptionalAction,
                        help="Save cropped images and masks")
    output.add_argument('--resize-crops', default=False, action=argparse.BooleanOptionalAction,
                        help="Resize crops to out-size (zooms in/out to match out-size). adds padding if necessary")
    output.add_argument('--preserve-crops', default=True, action=argparse.BooleanOptionalAction,
                        help="Keeps scale/resolution of cropped scenes. adds padding if necessary")
    output.add_argument('--flat', default=True, action=argparse.BooleanOptionalAction,
                        help="Store frames of all videos in output root directory instead of using sub directories per clip")
    output.add_argument('--save-as-images', default=False, action=argparse.BooleanOptionalAction,
                        help="Save as images instead of videos")

    nsfw_detection = parser.add_argument_group('NSFW detection')
    nsfw_detection.add_argument('--model', type=str, default="model_weights/lada_nsfw_detection_model.pt",
                        help="path to NSFW detection model")
    nsfw_detection.add_argument('--model-device', type=str, default="cuda", help="device to run the YOLO model on. E.g. 'cuda' or 'cuda:0'")

    scene_duration_filter = parser.add_argument_group('Scene duration filter')
    scene_duration_filter.add_argument('--scene-min-length', type=int, default=2.,
                        help="minimal length of a scene in number of frames in order to be detected (in seconds)")
    scene_duration_filter.add_argument('--scene-max-length', type=int, default=8,
                        help="maximum length of a scene in number of frames. Scenes longer than that will be cut (in seconds)")
    scene_duration_filter.add_argument('--scene-max-memory', default=6144, type=int, help="limits maximum scene length based on approximate memory consumption of the scene. Value should be given in Megabytes (MB)")

    video_quality_evaluation = parser.add_argument_group('Scene video quality evaluation')
    video_quality_evaluation.add_argument('--add-video-quality-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will evaluate video quality and add its results to metadata")
    video_quality_evaluation.add_argument('--enable-video-quality-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled and scene quality is below scene-min-quality it will be skipped and not land in the dataset.")
    video_quality_evaluation.add_argument('--video-quality-model-device', type=str, default="cuda", help="device to run the video quality model on. E.g. 'cuda' or 'cuda:0'")
    video_quality_evaluation.add_argument('--min-video-quality', type=float, default=0.1,
                        help="minimum quality of a scene as determined by quality estimation model DOVER. Range between 0 and 1 were 1 is highest quality. If scene quality is below this threshold it will be skipped and not land in the dataset.")

    mosaic_creation = parser.add_argument_group('Mosaic creation')
    mosaic_creation.add_argument('--save-mosaic', default=False, action=argparse.BooleanOptionalAction,
                        help="Create and save mosaic images and masks")
    mosaic_creation.add_argument('--degrade-mosaic', default=False, action=argparse.BooleanOptionalAction,
                        help="degrades mosaic and NSFW video clips to better match real world video sources (e.g. video compression artifacts)")

    watermark_detection = parser.add_argument_group('Watermark detection')
    watermark_detection.add_argument('--add-watermark-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will run watermark detection and add its results to metadata")
    watermark_detection.add_argument('--enable-watermark-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled, scenes obstructed by watermarks (arbitrary text or logos) will be skipped")
    watermark_detection.add_argument('--watermark-model-path', type=str, default="model_weights/lada_watermark_detection_model.pt",
                        help="path to watermark detection model")

    nsfw_detection = parser.add_argument_group('NudeNet NSFW detection')
    nsfw_detection.add_argument('--add-nudenet-nsfw-metadata', default=True, action=argparse.BooleanOptionalAction, help="If enabled will run NudeNet NSFW detection and add its results to metadata")
    nsfw_detection.add_argument('--enable-nudenet-nsfw-filter', default=False, action=argparse.BooleanOptionalAction, help="If enabled, scenes which aren't also classified by NudeNet as NSFW will be skipped")
    nsfw_detection.add_argument('--nudenet-nsfw-model-path', type=str, default="model_weights/3rd_party/640m.pt",
                        help="path to NudeNet NSFW detection model")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if not (args.save_cropped or args.save_uncropped):
        print("No save option given. Specify at least one!")
        return

    io_executor = ThreadPoolExecutor(max_workers=4)
    scenes_executor = ThreadPoolExecutor(max_workers=args.workers)

    nsfw_detection_model = YOLO(args.model)

    if args.add_video_quality_metadata or args.enable_video_quality_filter:
        video_quality_evaluator = VideoQualityEvaluator(device=args.video_quality_model_device)
    else:
        video_quality_evaluator = None

    if args.add_watermark_metadata or args.enable_watermark_filter:
        watermark_detector = WatermarkDetector(YOLO(args.watermark_model_path), device=args.model_device)
    else:
        watermark_detector = None

    if args.add_nudenet_nsfw_metadata or args.enable_nudenet_nsfw_filter:
        nudenet_nsfw_detector = NudeNetNsfwDetector(YOLO(args.nudenet_nsfw_model_path), device=args.model_device)
    else:
        nudenet_nsfw_detector = None

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
                                                          save_as_images=args.save_as_images,
                                                          quality_evaluation=VideoQualityProcessingOptions(args.enable_video_quality_filter, args.add_video_quality_metadata, args.min_video_quality),
                                                          watermark_detection=WatermarkDetectionProcessingOptions(args.enable_watermark_filter, args.add_watermark_metadata),
                                                          nudenet_nsfw_detection=NudeNetNsfwDetectionProcessingOptions(args.enable_nudenet_nsfw_filter, args.add_nudenet_nsfw_metadata))

        file_processing_options = FileProcessingOptions(scene_max_length=scene_max_length,
                                                        scene_min_length=args.scene_min_length,
                                                        stride_length=args.stride_length)

        scene_processing_futures.extend(process_file(nsfw_detection_model, video_metadata, output_dir, scenes_executor,
                     io_executor, video_quality_evaluator, watermark_detector, nudenet_nsfw_detector, file_processing_options, scene_processing_options, args.workers,  args.model_device))
        clean_up_completed_futures(scene_processing_futures)

    wait(scene_processing_futures, return_when=ALL_COMPLETED)

    io_executor.shutdown(wait=True)
    scenes_executor.shutdown(wait=True)


if __name__ == '__main__':
    main()
