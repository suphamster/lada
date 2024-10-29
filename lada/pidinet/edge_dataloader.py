import glob

import torch
from torch.utils import data
import os
import numpy as np
import cv2

class MosaicEdgesDataset(data.Dataset):
    def __init__(self, root, split):
        self.mosaics = sorted(glob.glob(os.path.join(root, split, 'mosaic', '*')))
        self.edges = sorted(glob.glob(os.path.join(root, split, 'mosaic_edge', '*')))

        rng = np.random.default_rng()
        state = rng.bit_generator.state
        rng.shuffle(self.mosaics)
        rng.bit_generator.state = state
        rng.shuffle(self.edges)

        self.mosaics = self.mosaics[:10000]
        self.edges = self.edges[:10000]

        self.split = split


    def __len__(self):
        return len(self.mosaics)

    def __getitem__(self, idx):
        mosaic = cv2.imread(self.mosaics[idx])
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        mosaic = mosaic.transpose(2, 0, 1)
        mosaic = torch.from_numpy(mosaic)
        mosaic = mosaic.float() / 255.0

        mosaic_filename = os.path.basename(self.mosaics[idx])

        edge = cv2.imread(self.edges[idx])
        edge = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        edge = np.expand_dims(edge, axis=0)
        edge = torch.from_numpy(edge)
        edge = edge.float() / 255.0

        assert mosaic.ndim == 3 and edge.ndim == 3

        return (mosaic, edge) if self.split == 'train' else (mosaic, mosaic_filename)