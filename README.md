## FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images

![Alt text](imgs/image.png)

This is the official code repository for "FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images".

This work is built upon **SAM2-PATH** [arxiv](https://arxiv.org/abs/2408.03651), which is a better segment anything model for semantic segmentation in digital pathology. We extend SAM2-PATH with frequency-aware mechanisms and prompt-guided strategies specifically designed for nuclear segmentation and classification in immunohistochemistry (IHC) images.

### Base Work

This code is based on:
- **SAM2-PATH** [git link](https://github.com/cvlab-stonybrook/SAMPath) - A better segment anything model for semantic segmentation in digital pathology
- **SAM-PATH** [Miccai conference paper](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_16) - The original SAM-PATH work
- **SAM2** [code](https://github.com/facebookresearch/segment-anything-2) - Meta's Segment Anything Model 2
- **UNI encoder** [git link](https://github.com/mahmoodlab/UNI) - Universal encoder for pathology images

All UNI and SAM2 pretrained weights can be downloaded from their respective repositories. Thanks to the authors for their excellent base code.