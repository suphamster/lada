from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .mosaic_video_dataset import MosaicVideoDataset

def create_dataloader(opt, phase, idx):
    # set params
    data_opt = opt['dataset'].get(idx)
    degradation_type = opt['dataset']['degradation']['type']

    # === create loader for training === #
    if phase == 'train':
        # check dataset
        assert data_opt['name'] in ('MosaicRemoval'), \
            f'Unknown Dataset: {data_opt["name"]}'

        if degradation_type == 'mosaic':
            dataset = MosaicVideoDataset(data_opt)
        else:
            raise ValueError(f'Unrecognized degradation type: {degradation_type}')

        # create data loader
        if opt['dist']:
            batch_size = data_opt['batch_size_per_gpu']
            shuffle = False
            sampler = DistributedSampler(dataset)
        else:
            batch_size = data_opt['batch_size_per_gpu']
            shuffle = True
            sampler = None

        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            sampler=sampler,
            num_workers=data_opt['num_worker_per_gpu'],
            pin_memory=data_opt['pin_memory'])

    # === create loader for testing === #
    elif phase == 'test':
        if degradation_type == 'mosaic':
            dataset = MosaicVideoDataset(data_opt)
        else:
            raise ValueError(f'Unrecognized degradation type: {degradation_type}')

        loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_worker_per_gpu'],
            pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError(f'Unrecognized phase: {phase}')

    return loader
