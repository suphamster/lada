import math
from pathlib import Path
from typing import Generator

import numpy as np

from lada.lib import Mask, Image, Box
from lada.lib import mask_utils
from lada.lib.scene_utils import crop_to_box_v3
from lada.lib.clean_frames_generator import CleanFrame, CleanFramesGenerator

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


class SceneGenerator:
    def __init__(self, yolo_object_generator: CleanFramesGenerator, scene_min_length: int, scene_max_length: int, random_extend_masks=False, stride_length=0):
        self.clean_frames_generator: CleanFramesGenerator = yolo_object_generator
        self.video_meta_data = yolo_object_generator.video_meta_data
        self.scene_min_length =  int(math.ceil(scene_min_length * self.video_meta_data.video_fps))
        self.scene_max_length = int(math.ceil(scene_max_length * self.video_meta_data.video_fps))
        self.total_frame_count = self.video_meta_data.frames_count
        self.video_file = self.video_meta_data.video_file
        self.random_extend_masks = random_extend_masks
        self.stride_length_frames = stride_length * self.video_meta_data.video_fps
        self.previous_scene_frame_end = None

    def __call__(self) -> Generator[Scene, None, None]:
        scene: Scene | None = None
        scenes_counter: int = 0

        clean_frame: CleanFrame
        for clean_frame in self.clean_frames_generator():
            scene_completed = False
            skip_scene = False

            if clean_frame.object_detected:
                if scene is None:
                    scene = Scene(self.video_file, clean_frame.object_id, self.scene_min_length, self.scene_max_length, self.video_meta_data)
                    scene.add_frame(clean_frame.frame_number, clean_frame.frame, clean_frame.mask,
                                    clean_frame.box)
                else:
                    if scene.id == clean_frame.object_id and scene.frame_end + 1 == clean_frame.frame_number:
                        scene.add_frame(clean_frame.frame_number, clean_frame.frame, clean_frame.mask,
                                        clean_frame.box)
                    else:
                        # todo: we'll lose/skip the current frame this way, we should mark the running scene as completed but also create a new scene with the the current frame
                        scene_completed = True
            elif scene is not None:
                scene_completed = True

            if scene is not None and clean_frame.frame_number == self.total_frame_count - 1:
                scene_completed = True

            skip_scene = not (scene_completed and scene.min_length_reached() and (self.previous_scene_frame_end is None or (scene.frame_start - self.previous_scene_frame_end) > self.stride_length_frames))

            if not skip_scene:
                scenes_counter += 1
                scene_complete = scene
                scene_complete.id = scenes_counter
                self.previous_scene_frame_end = scene.frame_end
                if self.random_extend_masks:
                    apply_random_mask_extensions(scene_complete)
                yield scene_complete
            if scene_completed:
                scene = None
                scene_complete = None