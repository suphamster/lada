from abc import ABC
from dataclasses import dataclass, asdict
import json
from typing import Optional

@dataclass
class MosaicMetadataV1:
    mod: str
    rect_ratio: float
    mosaic_size: int
    feather_size: float

@dataclass
class MosaicBlockSizeV1:
    mosaic_size_v2: float
    mosaic_size_v1_normal: float
    mosaic_size_v1_bounding: float

@dataclass
class VisualQualityScoreV1:
    aesthetic: float
    technical: float
    overall: float

@dataclass
class AbstractRestorationDatasetMetadata(ABC):
    def __init__(self):
        self.version = None

    def read_metadata_version(path: str) -> int:
        with open(path, 'r', encoding='utf-8') as f:
            json_dict = json.load(f)
        return json_dict['version'] if 'version' in json_dict else 1

    def to_json_file(self, path) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            json_dict = asdict(self)
            json.dump(json_dict, f)

    def from_json_file(path: str):
        raise NotImplementedError()

@dataclass
class RestorationDatasetMetadataV1(AbstractRestorationDatasetMetadata):
    version = 1
    fps: int
    frames_count: Optional[int]
    name: str
    orig_width: int
    orig_height: int
    base_mosaic_block_size: Optional[MosaicBlockSizeV1]
    mosaic: Optional[MosaicMetadataV1]
    pad: Optional[list[int]]
    height: int
    width: int
    video_quality: VisualQualityScoreV1
    frame_count: Optional[int] = None # deprecated

    def from_json_file(path: str):
        with open(path, 'r') as f:
            json_dict = json.load(f)
        version = json_dict['version'] if json_dict.get('version') else 1
        if version == 1:
            return RestorationDatasetMetadataV1(
                json_dict["fps"],
                json_dict.get("frames_count"),
                json_dict["name"],
                json_dict["orig_width"],
                json_dict["orig_height"],
                MosaicBlockSizeV1(
                    mosaic_size_v2=json_dict["base_mosaic_block_size"]["mosaic_size_v2"],
                    mosaic_size_v1_normal=json_dict["base_mosaic_block_size"]["mosaic_size_v1_normal"],
                    mosaic_size_v1_bounding=json_dict["base_mosaic_block_size"]["mosaic_size_v1_bounding"],
                ) if json_dict.get("base_mosaic_block_size") else None,
                MosaicMetadataV1(
                    mod=json_dict["mosaic"]["mod"],
                    rect_ratio=json_dict["mosaic"]["rect_ratio"],
                    mosaic_size=json_dict["mosaic"]["mosaic_size"],
                    feather_size=json_dict["mosaic"]["feather_size"],
                ) if json_dict.get('mosaic') else None,
                json_dict.get("pad"),
                json_dict["height"],
                json_dict["width"],
                VisualQualityScoreV1(
                    json_dict["video_quality"]["aesthetic"],
                    json_dict["video_quality"]["technical"],
                    json_dict["video_quality"]["overall"],
                ),
                json_dict.get("frame_count"),
            )
