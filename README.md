## FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images

![Alt text](imgs/image.png)

This is the official code repository for "FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images".

### Introduction

Accurate nuclear segmentation and classification (NuSC) in immunohistochemistry (IHC)-stained images is essential for reliable biomarker quantification, yet existing methods frequently underperform due to domain-specific challenges such as stain heterogeneity and low nuclear contrast. FPNuNet addresses these limitations through a frequency-aware prompt-guided architecture that integrates RGB and Hematoxylin-Eosin-Diaminobenzidine channels via a color fusion stem, processes features through SAM-based structural and ViT-based semantic encoders modulated by lightweight prompt adapters, and employs a progressive frequency-aware residual global fusion neck to aggregate multi-scale features. The network utilizes three collaborative decoder branches to jointly predict binary masks, horizontal-vertical vectors, and nuclear types, enabling robust performance across diverse IHC staining conditions. Experimental evaluation on the CD47-IHCNuSC dataset demonstrates that FPNuNet consistently outperforms state-of-the-art baselines in both segmentation accuracy and classification robustness.

This work is built upon **SAM2-PATH** [arxiv](https://arxiv.org/abs/2408.03651), which is a better segment anything model for semantic segmentation in digital pathology. We extend SAM2-PATH with frequency-aware mechanisms and prompt-guided strategies specifically designed for nuclear segmentation and classification in immunohistochemistry (IHC) images.

### Base Work

This code is based on:
- **SAM2-PATH** [git link](https://github.com/cvlab-stonybrook/SAMPath) - A better segment anything model for semantic segmentation in digital pathology
- **SAM-PATH** [Miccai conference paper](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_16) - The original SAM-PATH work
- **SAM2** [code](https://github.com/facebookresearch/segment-anything-2) - Meta's Segment Anything Model 2
- **UNI encoder** [git link](https://github.com/mahmoodlab/UNI) - Universal encoder for pathology images

All UNI and SAM2 pretrained weights can be downloaded from their respective repositories. Thanks to the authors for their excellent base code.