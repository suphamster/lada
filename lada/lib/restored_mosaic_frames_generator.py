from ultralytics import YOLO

from lada.lib import image_utils, video_utils
from lada.lib import visualization
from lada.lib.mosaic_frames_generator import MosaicFramesGenerator
from lada.lib.clean_mosaic_utils import clean_cropped_mosaic
from lada.pidinet import pidinet_inference


def load_models(device, mosaic_restoration_model_name, mosaic_restoration_model_path, mosaic_restoration_config_path,
                mosaic_detection_model_path, mosaic_cleaning_edge_detection_model_path=None):
    mosaic_edge_detection_model = None
    if mosaic_cleaning_edge_detection_model_path:
        mosaic_edge_detection_model = pidinet_inference.load_model(mosaic_cleaning_edge_detection_model_path, model_type="tiny")

    if mosaic_restoration_model_name == "rvrt":
        from lada.rvrt import rvrt_inferencer
        mosaic_restoration_model = rvrt_inferencer.get_model(model_path=mosaic_restoration_model_path, device=device)
        pad_mode = 'zero'
    elif mosaic_restoration_model_name == "deepmosaics":
        from lada.deepmosaics.models import loadmodel, model_util
        mosaic_restoration_model = loadmodel.video(model_util.device_to_gpu_id(device), mosaic_restoration_model_path)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name.startswith("basicvsrpp"):
        from lada.basicvsrpp.inference import load_model, get_default_gan_inference_config
        if mosaic_restoration_config_path:
            config = mosaic_restoration_config_path
        else:
            config = get_default_gan_inference_config()
        mosaic_restoration_model = load_model(config, mosaic_restoration_model_path, device)
        pad_mode = 'reflect'
    elif mosaic_restoration_model_name == "tecogan":
        from lada.tecogan.tecogan_inferencer import load_model
        mosaic_restoration_model = load_model(mosaic_restoration_config_path)
        pad_mode = 'reflect'
    else:
        raise NotImplementedError()

    mosaic_detection_model = YOLO(mosaic_detection_model_path)
    return mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, pad_mode


class FrameRestorer:
    def __init__(self, device, video_file, preserve_relative_scale, max_clip_length, mosaic_restoration_model_name,
                 mosaic_detection_model, mosaic_restoration_model, mosaic_edge_detection_model, preferred_pad_mode,
                 start_ns=0, passthrough=False, mosaic_detection=False, mosaic_cleaning=False):
        self.device = device
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.max_clip_length = max_clip_length
        self.preserve_relative_scale = preserve_relative_scale
        self.video_file = video_file
        self.mosaic_detection_model = mosaic_detection_model
        self.mosaic_restoration_model = mosaic_restoration_model
        self.mosaic_edge_detection_model = mosaic_edge_detection_model
        self.preferred_pad_mode = preferred_pad_mode
        self.start_ns = start_ns
        self.passthrough = passthrough
        self.mosaic_cleaning = mosaic_cleaning
        self.mosaic_detection = mosaic_detection

    def restore_clip(self, images):
        if self.mosaic_restoration_model_name == "rvrt":
            from lada.rvrt import rvrt_inferencer
            restored_clip_images = rvrt_inferencer.inference(images, self.mosaic_restoration_model)
        elif self.mosaic_restoration_model_name == "deepmosaics":
            from lada.deepmosaics.inference import restore_video_frames
            from lada.deepmosaics.models import model_util
            restored_clip_images = restore_video_frames(model_util.device_to_gpu_id(self.device), self.mosaic_restoration_model, images)
        elif self.mosaic_restoration_model_name.startswith("basicvsrpp"):
            from lada.basicvsrpp.inference import inference
            restored_clip_images = inference(self.mosaic_restoration_model, images, self.device)
        elif self.mosaic_restoration_model_name == "tecogan":
            from lada.tecogan.tecogan_inferencer import inference
            restored_clip_images = inference(images, self.mosaic_restoration_model)
        else:
            raise NotImplementedError()
        return restored_clip_images

    def __call__(self):
        with video_utils.VideoReader(self.video_file) as video_reader:
            video_metadata = video_utils.get_video_meta_data(self.video_file)
            frame_num = video_utils.offset_ns_to_frame_num(self.start_ns, video_metadata.video_fps_exact)
            if self.start_ns > 0:
                video_reader.seek(self.start_ns)

            video_frames_generator = video_reader.frames()

            if self.passthrough:
                for frame, frame_pts in video_frames_generator:
                    yield frame, frame_pts
                    frame_num += 1

            else:
                mosaic_generator = MosaicFramesGenerator(self.mosaic_detection_model, self.video_file, max_clip_length=self.max_clip_length,
                                                         pad_mode=self.preferred_pad_mode, preserve_relative_scale=self.preserve_relative_scale,
                                                         dont_preserve_relative_scale=(not self.preserve_relative_scale), start_ns=self.start_ns)
                clip_buffer = []

                for clip_idx, clip in enumerate(mosaic_generator()):
                    if self.mosaic_detection:
                        restored_clip_images = visualization.draw_mosaic_detections(clip)
                    else:
                        if self.mosaic_cleaning:
                            images = []
                            for (cropped_img, cropped_mask, cropped_box, orig_crop_shape, pad_after_resize, pad_before_resize) in clip:
                                images.append(clean_cropped_mosaic(cropped_img, cropped_mask, pad_after_resize, pidinet_model=self.mosaic_edge_detection_model))
                        else:
                            images = clip.get_clip_images()

                        restored_clip_images = self.restore_clip(images)
                    assert len(restored_clip_images) == len(clip.get_clip_images())

                    for i in range(len(restored_clip_images)):
                        assert clip.data[i][0].shape == restored_clip_images[i].shape
                        clip.data[i] = restored_clip_images[i], clip.data[i][1], clip.data[i][2], clip.data[i][3], clip.data[i][4], clip.data[i][5]

                    if clip.frame_start > frame_num:
                        if len(clip_buffer) == 0:
                            while clip.frame_start > frame_num:
                                frame, frame_pts = next(video_frames_generator)
                                yield frame, frame_pts
                                frame_num += 1
                        else:
                            while True:
                                if len(clip_buffer) == 0 or clip.frame_start <= min(clip_buffer, key=lambda c: c.frame_start).frame_start:
                                    break
                                frame, frame_pts = next(video_frames_generator)
                                for buffered_clip in [c for c in clip_buffer if c.frame_start == frame_num]:
                                    clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize, pad_before_resize = buffered_clip.pop()
                                    clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
                                    clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
                                    clip_img = image_utils.unpad_image(clip_img, pad_before_resize)
                                    t, l , b, r = orig_clip_box
                                    frame[t:b + 1, l:r + 1, :] = clip_img
                                yield frame, frame_pts
                                frame_num += 1

                                processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
                                for processed_clip in processed_clips:
                                    clip_buffer.remove(processed_clip)

                    clip_buffer.append(clip)

                for frame, frame_pts in video_frames_generator:
                    if len(clip_buffer) == 0:
                        yield frame, frame_pts
                        frame_num += 1
                    else:
                        for buffered_clip in [c for c in clip_buffer if c.frame_start == frame_num]:
                            clip_img, clip_mask, orig_clip_box, orig_crop_shape, pad_after_resize, pad_before_resize = buffered_clip.pop()
                            clip_img = image_utils.unpad_image(clip_img, pad_after_resize)
                            clip_img = image_utils.resize(clip_img, orig_crop_shape[:2])
                            clip_img = image_utils.unpad_image(clip_img, pad_before_resize)
                            t, l, b, r = orig_clip_box
                            frame[t:b + 1, l:r + 1, :] = clip_img
                        yield frame, frame_pts
                        frame_num += 1

                        processed_clips = list(filter(lambda _clip: len(_clip) == 0, clip_buffer))
                        for processed_clip in processed_clips:
                            clip_buffer.remove(processed_clip)
