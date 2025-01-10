import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import ultralytics.models

from lada.lib import Mask, Image, Box, VideoMetadata
from lada.lib import mask_utils
from lada.lib.scene_utils import crop_to_box_v3
from lada.lib.ultralytics_utils import choose_biggest_detection, convert_yolo_mask, convert_yolo_box


@dataclass
class NsfwFrame:
    frame_number: int
    last_frame: bool
    frame: Image
    _box: ultralytics.engine.results.Boxes
    _mask: ultralytics.engine.results.Masks
    object_detected: bool = False
    object_id: int = None

    @property
    def mask(self) -> Mask:
        mask = convert_yolo_mask(self._mask, self.frame.shape)
        mask = mask_utils.fill_holes(mask)
        return mask

    @property
    def box(self) -> Box:
        return convert_yolo_box(self._box, self.frame.shape)


class Scene:

    def __init__(self, file_path: Path, id, scene_min_length, scene_max_length, video_meta_data):
        self.file_path = file_path
        self.video_meta_data = video_meta_data
        self.id: int = id
        self.data: list = []
        self.frame_start: int | None = None
        self.frame_end: int | None = None
        self._index: int = 0
        self.scene_max_length: int = scene_max_length
        self.scene_min_length: int = scene_min_length

    def __len__(self):
        return len(self.data)

    def min_length_reached(self):
        return len(self) >= self.scene_min_length

    def max_length_reached(self):
        return len(self) >= self.scene_max_length

    def add_frame(self, frame_num, img, mask, box):
        if self.frame_start is None:
            self.frame_start = frame_num
            self.frame_end = frame_num
            self.data.append((img, mask, box))
        elif not self.max_length_reached():
            assert frame_num == self.frame_end + 1
            self.frame_end = frame_num
            self.data.append((img, mask, box))

    def get_images(self) -> list[Image]:
        return [img for img, _, _ in self.data]

    def get_masks(self) -> list[Mask]:
        return [mask for _, mask, _ in self.data]

    def get_boxes(self) -> list[Box]:
        return [box for _, _, box in self.data]

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, Mask, Box]:
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item) -> tuple[Image, Mask, Box]:
        return self.data[item]


class CroppedScene:

    def __init__(self, scene: Scene, window_in_seconds=1.0, target_size=(400,400), smoothing=True, border_size=0):
        self.file_path = scene.file_path
        self.id: int = scene.id
        self.data: list = []
        self._index: int = 0

        if smoothing:
            smoothed_boxes = SmoothSceneBoxes.smooth_boxes(scene, window_in_seconds, smooth_function='median')
        else:
            smoothed_boxes = scene.get_boxes()
        scene_images = scene.get_images()
        scene_mask_images = scene.get_masks()

        for i, smoothed_box in enumerate(smoothed_boxes):
            cropped_image, cropped_mask_image, cropped_box, scale_factor = crop_to_box_v3(smoothed_box, scene_images[i],
                                                                         scene_mask_images[i], target_size, border_size=border_size)
            self.data.append((cropped_image, cropped_mask_image, cropped_box))

    def __len__(self):
        return len(self.data)

    def get_images(self) -> list[Image]:
        return [img for img, _, _ in self.data]

    def get_masks(self) -> list[Mask]:
        return [mask for _, mask, _ in self.data]

    def get_boxes(self) -> list[Box]:
        """
        Location of cropped area in original image
        """
        return [box for _, _, box in self.data]

    def get_max_width_height(self):
        max_width = 0
        max_height = 0
        for _, _, box in self.data:
            t, l, b, r = box
            width, height = r - l + 1, b - t + 1
            if height > max_height:
                max_height = height
            if width > max_width:
                max_width = width
        return max_width, max_height

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Image, Mask, Box]:
        if self._index < len(self):
            item = self.data[self._index]
            self._index += 1
            return item
        else:
            raise StopIteration

    def __getitem__(self, item) -> tuple[Image, Mask, Box]:
        return self.data[item]


class SmoothSceneBoxes:

    @staticmethod
    def median_filter(data, window=11):
        assert window % 2 != 0
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        data_type = data[0]

        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros(data_size, dtype=data_type)
        for i in range(data_size):
            filtered_data[i] = np.median(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def mean_filter(data, window=11):
        assert window % 2 != 0
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        data_type = data[0]
        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros(data_size, dtype=data_type)
        for i in range(data_size):
            filtered_data[i] = np.mean(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def min_max_filter(data, window, mode):
        assert window % 2 != 0
        func = np.max if mode == 'max' else np.min
        pad_size = int((window - 1) / 2)
        data_size = len(data)
        padded_data = np.pad(data, pad_size, 'edge')
        filtered_data = np.zeros_like(data)
        for i in range(data_size):
            filtered_data[i] = func(padded_data[i:i + window])
        return filtered_data

    @staticmethod
    def smooth_boxes(scene: Scene, window_in_seconds: float, smooth_function='median'):
        _scene_boxes = np.array(scene.get_boxes())
        window_in_frames = min(math.ceil(window_in_seconds * scene.video_meta_data.video_fps), len(scene))
        if window_in_frames % 2 == 0:
            window_in_frames -= 1
        if window_in_frames < 1:
            return _scene_boxes.tolist()

        for i, position in zip(range(4),('t','l','b','r')):
            if smooth_function == 'median':
                _scene_boxes[:, i] = SmoothSceneBoxes.median_filter(_scene_boxes[:, i], window_in_frames)
            elif smooth_function == 'min_max':
                _scene_boxes[:, i] = SmoothSceneBoxes.min_max_filter(_scene_boxes[:, i], window_in_frames, 'max' if position in ('b', 'r') else 'min')
            elif smooth_function == 'mean':
                _scene_boxes[:, i] = SmoothSceneBoxes.mean_filter(_scene_boxes[:, i], window_in_frames)
            else:
                raise NotImplementedError()

        return _scene_boxes.tolist()

    @staticmethod
    def smooth_boxes_center_point(scene: Scene, window_in_seconds: float, smooth_function='median'):
        _scene_boxes = np.array(scene.get_boxes())

        window_in_frames = min(math.ceil(window_in_seconds * scene.video_meta_data.video_fps), len(scene))
        if window_in_frames % 2 == 0:
            window_in_frames -= 1

        heights = _scene_boxes[:, 2] - _scene_boxes[:, 0]
        widths = _scene_boxes[:, 3] - _scene_boxes[:, 1]

        half_widths = widths / 2
        half_heights = heights / 2

        if smooth_function == 'median':
            smoothed_heights = SmoothSceneBoxes.median_filter(heights, window_in_frames)
            smoothed_widths = SmoothSceneBoxes.median_filter(widths, window_in_frames)
            center_x = SmoothSceneBoxes.median_filter(_scene_boxes[:, 1] + half_widths, window_in_frames)
            center_y = SmoothSceneBoxes.median_filter(_scene_boxes[:, 0] + half_heights, window_in_frames)
        else:
            raise NotImplementedError()

        half_smoothed_widths = smoothed_widths / 2
        half_smoothed_heights = smoothed_heights / 2

        new_t = np.clip((center_y - half_smoothed_heights).round().astype(np.int64), 0, scene.video_meta_data.video_height)
        new_b = np.clip((center_y + half_smoothed_heights).round().astype(np.int64), 0, scene.video_meta_data.video_height)
        new_l = np.clip((center_x - half_smoothed_widths).round().astype(np.int64), 0, scene.video_meta_data.video_width)
        new_r = np.clip((center_x + half_smoothed_widths).round().astype(np.int64), 0, scene.video_meta_data.video_width)

        smoothed_boxes = np.stack((new_t, new_l, new_b, new_r), axis=-1)
        return smoothed_boxes.tolist()

def apply_random_mask_extensions(scene: Scene):
    value = np.random.choice([0, 0, 1, 1, 2])
    for i in range(len(scene)):
        img, mask, box = scene.data[i]
        mask_extended = mask_utils.extend_mask(mask, value)
        box_extended = mask_utils.get_box(mask_extended)
        scene.data[i] = img, mask_extended, box_extended

class NsfwFramesGenerator:
    def __init__(self, model: ultralytics.models.YOLO, video_meta_data: VideoMetadata, device=None):
        self.model = model
        self.device = torch.device(device) if device is not None else device
        self.video_meta_data = video_meta_data

    def __call__(self, *args, **kwargs) -> Generator[NsfwFrame, None, None]:
        nsfw_frame = None
        for frame_num, results in enumerate(self.model.track(source=self.video_meta_data.video_file, stream=True, verbose=False, tracker="bytetrack.yaml", device=self.device)):
            if nsfw_frame:
                yield nsfw_frame
            yolo_box, yolo_mask = choose_biggest_detection(results, tracking_mode=True)
            object_detected = yolo_box is not None
            nsfw_frame = NsfwFrame(frame_num, False, results.orig_img, yolo_box, yolo_mask, object_detected, int(yolo_box.id.item()) if object_detected else None)
        if nsfw_frame: # EOF
            nsfw_frame.last_frame = True
            yield nsfw_frame

class NsfwSceneGenerator:
    def __init__(self, model: ultralytics.models.YOLO, video_meta_data: VideoMetadata, device, scene_min_length: int, scene_max_length: int, random_extend_masks=False, stride_length=0):
        self.nsfw_frames_generator: NsfwFramesGenerator = NsfwFramesGenerator(model, video_meta_data, device)
        self.video_meta_data = video_meta_data
        self.scene_min_length =  int(math.ceil(scene_min_length * self.video_meta_data.video_fps))
        self.scene_max_length = int(math.ceil(scene_max_length * self.video_meta_data.video_fps))
        self.video_file = self.video_meta_data.video_file
        self.random_extend_masks = random_extend_masks
        self.stride_length_frames = stride_length * self.video_meta_data.video_fps
        self.previous_completed_scene_frame_end = None
        self.scenes_counter: int = 0

    def _process_completed_scene(self, completed_scene: Scene) -> Optional[Scene]:
        """returns Scene if it fits the criteria for a valid completed scene like min/max length"""
        # todo: implementing video strides / discarding frames here is inefficient as we still pass every frame through YOLO / nsfw frames generator.
        #  We'd need to read and pass only the frames we want to be processed to YOLO instead of passing the video file.
        skip_scene = not (completed_scene.min_length_reached() and (self.previous_completed_scene_frame_end is None or (completed_scene.frame_start - self.previous_completed_scene_frame_end) > self.stride_length_frames))
        if skip_scene:
            return None
        self.scenes_counter += 1
        completed_scene.id = self.scenes_counter
        self.previous_completed_scene_frame_end = completed_scene.frame_end
        if self.random_extend_masks:
            apply_random_mask_extensions(completed_scene)
        return completed_scene

    def __call__(self) -> Generator[Scene, None, None]:
        scene: Scene | None = None
        nsfw_frame: NsfwFrame

        for nsfw_frame in self.nsfw_frames_generator():
            if nsfw_frame.object_detected:
                if scene is None:
                    scene = Scene(self.video_file, nsfw_frame.object_id, self.scene_min_length, self.scene_max_length, self.video_meta_data)
                    scene.add_frame(nsfw_frame.frame_number, nsfw_frame.frame, nsfw_frame.mask, nsfw_frame.box)
                else:
                    if scene.id == nsfw_frame.object_id and scene.frame_end + 1 == nsfw_frame.frame_number:
                        scene.add_frame(nsfw_frame.frame_number, nsfw_frame.frame, nsfw_frame.mask, nsfw_frame.box)
                    else:
                        completed_scene = self._process_completed_scene(scene)
                        if completed_scene:
                            yield completed_scene
                        scene = Scene(self.video_file, nsfw_frame.object_id, self.scene_min_length, self.scene_max_length, self.video_meta_data)
                        scene.add_frame(nsfw_frame.frame_number, nsfw_frame.frame, nsfw_frame.mask, nsfw_frame.box)

            if scene is not None and (nsfw_frame.last_frame or not nsfw_frame.object_detected):
                completed_scene = self._process_completed_scene(scene)
                if completed_scene:
                    yield completed_scene
                scene = None
