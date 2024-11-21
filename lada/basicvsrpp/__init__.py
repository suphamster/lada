def register_all_modules():
    from lada.basicvsrpp.mmagic import register_all_modules
    register_all_modules()
    from lada.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet, BasicVSRPlusPlusGan
    from lada.basicvsrpp.mosaic_video_dataset import MosaicVideoDataset