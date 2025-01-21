Code vendored from MMCV, published under Apache License
https://github.com/open-mmlab/mmcv/blob/main/LICENSE

Only kept part used by BasicVSR++ and RealBasicVSR training code. Issue with MMCV was that its currently not well maintained which makes it hard to install/package.
It was mainly used by BasicVSR++ for its DeformConv operator CUDA implementation. Instead, we now use the implementation provided by torchvision while vendoring/copy-pasting
the remaining use of pure Python code into the lada.basicvsrpp.mmcv package.