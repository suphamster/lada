import logging
from lada import LOG_LEVEL
from lada.lib import VideoMetadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=LOG_LEVEL)
from lada import RESTORATION_MODEL_NAMES_TO_FILES, DETECTION_MODEL_NAMES_TO_FILES
from lada.lib.frame_restorer import load_models, FrameRestorer
from lada.lib import video_utils

class FrameRestorerProvider:
    def __init__(self, mosaic_restoration_model_name, mosaic_detection_model_name, video_metadata: VideoMetadata,
                 device, max_clip_length, mosaic_detection, passthrough):
        self.models_cache: None | dict = None
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.mosaic_detection_model_name = mosaic_detection_model_name
        self.video_metadata = video_metadata
        self.device = device
        self.max_clip_length = max_clip_length
        self.mosaic_detection = mosaic_detection
        self.passthrough = passthrough

    def reinit(self, mosaic_restoration_model_name, mosaic_detection_model_name, video_metadata: VideoMetadata,
                 device, max_clip_length, mosaic_detection, passthrough):
        self.mosaic_restoration_model_name = mosaic_restoration_model_name
        self.mosaic_detection_model_name = mosaic_detection_model_name
        self.video_metadata = video_metadata
        if self.device != device:
            self.models_cache = None
        self.device = device
        self.max_clip_length = max_clip_length
        self.mosaic_detection = mosaic_detection
        self.passthrough = passthrough

    def get(self):
        if self.passthrough:
            return PassthroughFrameRestorer(self.video_metadata.video_file)

        is_empty_cache = self.models_cache is None
        cache_miss = False
        if is_empty_cache:
            cache_miss = True
        else:
            if self.models_cache["mosaic_restoration_model_name"] != self.mosaic_restoration_model_name:
                cache_miss = True
                logger.info(f"model {self.mosaic_restoration_model_name} not found in cache. Loading...")
            if self.models_cache["mosaic_detection_model_name"] != self.mosaic_detection_model_name:
                cache_miss = True
                logger.info(f"model {self.mosaic_detection_model_name} not found in cache. Loading...")

        if cache_miss:
            mosaic_restoration_model_path = RESTORATION_MODEL_NAMES_TO_FILES[self.mosaic_restoration_model_name]
            mosaic_detection_path = DETECTION_MODEL_NAMES_TO_FILES[self.mosaic_detection_model_name]
            mosaic_detection_model, mosaic_restoration_model, mosaic_restoration_model_preferred_pad_mode = load_models(
                self.device, self.mosaic_restoration_model_name, mosaic_restoration_model_path, None,
                mosaic_detection_path
            )

            self.models_cache = dict(mosaic_restoration_model_name=self.mosaic_restoration_model_name,
                                     mosaic_detection_model_name=self.mosaic_detection_model_name,
                                     mosaic_detection_model=mosaic_detection_model,
                                     mosaic_restoration_model=mosaic_restoration_model,
                                     mosaic_restoration_model_preferred_pad_mode=mosaic_restoration_model_preferred_pad_mode)

        return FrameRestorer(self.device, self.video_metadata.video_file, True, self.max_clip_length,
                             self.mosaic_restoration_model_name,
                             self.models_cache["mosaic_detection_model"], self.models_cache["mosaic_restoration_model"],
                             self.models_cache["mosaic_restoration_model_preferred_pad_mode"],
                             mosaic_detection=self.mosaic_detection)

class PassthroughFrameRestorer:
    def __init__(self, video_file):
        self.video_file = video_file
        self.video_reader: video_utils.VideoReader | None = None
        self.frame_restoration_queue = None
        self.stopped = False

    def start(self, start_ns=0):
        self.video_reader = video_utils.VideoReader(self.video_file)
        self.video_reader = self.video_reader.__enter__()
        if start_ns >= 0:
            self.video_reader.seek(start_ns)
        self.frame_restoration_queue = PassthroughFrameRestorer.PassthroughQueue(self)

    def stop(self):
        self.stopped = True
        self.video_reader.__exit__(None, None, None)

    def get_frame_restoration_queue(self):
        return self.frame_restoration_queue

    class PassthroughQueue:
        def __init__(self, frame_restorer):
            self.video_frames_generator = frame_restorer.video_reader.frames()
            self.frame_restorer = frame_restorer

        def get(self, block=True, timeout=None):
            if self.frame_restorer.stopped:
                return None
            try:
                return next(self.video_frames_generator)
            except StopIteration:
                return None

        def put(self, item, block=True, timeout=None):
            pass
