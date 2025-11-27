## FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images

![Alt text](imgs/image.png)

This is the official code repository for "FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images".

### Introduction

Accurate nuclear segmentation and classification (NuSC) in immunohistochemistry (IHC)-stained images is essential for reliable biomarker quantification, yet existing methods frequently underperform due to domain-specific challenges such as stain heterogeneity and low nuclear contrast. FPNuNet addresses these limitations through a frequency-aware prompt-guided architecture that integrates RGB and Hematoxylin-Eosin-Diaminobenzidine channels via a color fusion stem, processes features through SAM-based structural and ViT-based semantic encoders modulated by lightweight prompt adapters, and employs a progressive frequency-aware residual global fusion neck to aggregate multi-scale features. The network utilizes three collaborative decoder branches to jointly predict binary masks, horizontal-vertical vectors, and nuclear types, enabling robust performance across diverse IHC staining conditions. Experimental evaluation on the CD47-IHCNuSC dataset demonstrates that FPNuNet consistently outperforms state-of-the-art baselines in both segmentation accuracy and classification robustness.

This work is built upon **SAM2-PATH** [arxiv](https://arxiv.org/abs/2408.03651), which is a better segment anything model for semantic segmentation in digital pathology. We extend SAM2-PATH with frequency-aware mechanisms and prompt-guided strategies specifically designed for nuclear segmentation and classification in immunohistochemistry (IHC) images.

### Citation

If you use our code or data in your research, please cite our paper:

```bibtex
@article{qin2026fpnunet,
  title={FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear Segmentation and Classification in Immunohistochemistry Images},
  author={Lulu Qin, Zhigang Pei, Xudong He, Jiarui Zhou, Xianhong Xu and Zexuan Zhu},
  journal={GigaScience},
  year={2026}
}
```

---

## 🚀 Download and Install

### System Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended for training)
- CUDA Toolkit (for GPU support)
- PyTorch >= 1.13.0
- Sufficient disk space for datasets and model checkpoints

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/your_username/FPNuNet.git
cd FPNuNet
```

2. **Create a conda environment (recommended):**

```bash
conda create -n fpnunet python=3.8
conda activate fpnunet
```

3. **Install PyTorch:**

   Install PyTorch according to your CUDA version from [PyTorch official website](https://pytorch.org/). For example:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

   The `requirements.txt` includes all necessary packages:
   - Core deep learning frameworks (PyTorch, PyTorch Lightning)
   - Data processing libraries (NumPy, SciPy, h5py, OpenCV)
   - Configuration utilities (PyYAML, box)
   - Visualization tools (Matplotlib, imageio)
   - Machine learning tools (scikit-learn, timm)
   - Logging and monitoring (wandb, tensorboard)
   - Data augmentation (albumentations)
   - Metrics (torchmetrics)

5. **Download pretrained weights:**

   - **SAM2 weights**: Download from [SAM2 repository](https://github.com/facebookresearch/segment-anything-2)
     - Recommended: `sam_vit_b_01ec64.pth` (ViT-B model)
   - **UNI encoder weights**: Download from [UNI repository](https://github.com/mahmoodlab/UNI)
     - Download the PyTorch model file (e.g., `pytorch_model.bin`)

   Place the downloaded weights in appropriate directories and update the paths in the configuration files.

---

## ⚙️ Usage

### Data Preparation

1. **Download the CD47-IHCNuSC dataset:**

   The dataset can be found in the `CD47_IHCNUSC/` directory. For detailed dataset information, please refer to `CD47_IHCNUSC/readme_en.md`.
   
   **Note**: The CD47-IHCNuSC dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Please review the license file in `CD47_IHCNUSC/` before use.

2. **Prepare data patches:**

   The dataset should be preprocessed into HDF5 format. The expected data structure in HDF5 files includes:
   - Image patches
   - Instance segmentation masks
   - Nuclear type labels
   - Horizontal-vertical (HV) vectors
   
   Organize your processed data as specified in the configuration file:
   ```
   dataset_root/
   ├── gt_{name}_train_128x128_64x64_img_inst_type_hv.h5
   └── gt_{name}_valid_128x128_64x64_img_inst_type_hv.h5
   ```

3. **Update configuration file:**

   Edit the configuration file in `configs/` directory (e.g., `CD47_nuclei_HV_h5_128x128.py`) to set:
   - `dataset_root`: Path to your dataset directory
   - `checkpoint`: Path to SAM2 pretrained weights (e.g., `sam_vit_b_01ec64.pth`)
   - `extra_checkpoint`: Path to UNI encoder pretrained weights (e.g., `pytorch_model.bin`)
   - `out_dir`: Output directory for saving model checkpoints and logs
   - `batch_size`: Training batch size (default: 12)
   - `num_epochs`: Number of training epochs (default: 80)
   - `learning_rate`: Initial learning rate (default: 5e-4)
   - `devices`: GPU device IDs for training (e.g., `[0]` for single GPU, `[0, 1]` for multi-GPU)

### Training

Run the training script with your configuration file:

```bash
python main/main_FPNuNet.py --config configs/CD47_nuclei_HV_h5_128x128.py
```

#### Training Options

The training script supports various options that can be configured in the config file:

- `batch_size`: Training batch size (default: 12)
- `num_epochs`: Number of training epochs (default: 80)
- `learning_rate`: Initial learning rate (default: 5e-4)
- `devices`: GPU device IDs (e.g., [0, 1] for multi-GPU training)
- `out_dir`: Output directory for saving model checkpoints and logs

#### Training Example

```bash
# Single GPU training
python main/main_FPNuNet.py --config configs/CD47_nuclei_HV_h5_128x128.py

# Multi-GPU training (if supported)
# Set devices in config file: devices = [0, 1, 2, 3]
```

The training process will save model checkpoints and logs to the directory specified in the configuration file. You can monitor training progress using TensorBoard:

```bash
tensorboard --logdir <your_output_directory>/logs
```

---

## 📁 Project Structure

```
FPNuNet/
├── main/                      # Main training scripts
│   ├── main_FPNuNet.py        # Main training entry point
│   ├── trainer.py             # Training logic
│   ├── config_manager.py      # Configuration management
│   ├── utils.py               # Utility functions for data and model creation
│   ├── get_model.py           # Model initialization
│   ├── pl_module_multiHead.py # PyTorch Lightning module
│   ├── metrics.py             # Evaluation metrics
│   ├── infer_metrics.py       # Inference metrics (AJI, PQ, Dice, etc.)
│   ├── losses_v5.py           # Loss functions
│   ├── h5dataloder_v21.py     # HDF5 data loader
│   ├── multi_train.py         # Multi-model training script
│   └── ...
├── network/                   # Network architecture modules
│   ├── SAM_NuSCNet_v231.py    # Main network architecture
│   ├── sam_pfae_fusion_neck_modules.py  # Frequency-aware fusion modules
│   └── ...
├── configs/                   # Configuration files
│   └── CD47_nuclei_HV_h5_128x128.py
├── CD47_IHCNUSC/              # Dataset directory
│   ├── images/                # Image files
│   ├── annotations/           # Annotation files
│   └── readme_en.md           # Dataset documentation
├── sam2_train/                # SAM2 related modules
├── misc/                      # Miscellaneous utilities
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The CD47-IHCNuSC dataset is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Please refer to the license file in `CD47_IHCNUSC/` for details.

---

## 🙏 Acknowledgments

This code is based on:
- **SAM2-PATH** [git link](https://github.com/cvlab-stonybrook/SAMPath) - A better segment anything model for semantic segmentation in digital pathology
- **SAM-PATH** [Miccai conference paper](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_16) - The original SAM-PATH work
- **SAM2** [code](https://github.com/facebookresearch/segment-anything-2) - Meta's Segment Anything Model 2
- **UNI encoder** [git link](https://github.com/mahmoodlab/UNI) - Universal encoder for pathology images

All UNI and SAM2 pretrained weights can be downloaded from their respective repositories. Thanks to the authors for their excellent base code.
