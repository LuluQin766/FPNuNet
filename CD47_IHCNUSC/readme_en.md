# CD47-IHCNuSC Dataset Documentation

## Dataset Overview

**CD47-IHCNuSC** is a new benchmark dataset for biomarker-aware nuclear segmentation and classification (NuSC) under clinically realistic immunohistochemistry (IHC) conditions. The dataset focuses on CD47-stained esophageal cancer tissue, providing expert-validated instance masks with seven fine-grained, phenotype-aware nuclear subtypes.

To enable rigorous evaluation of biomarker-aware NuSC under clinically realistic IHC conditions, we constructed CD47-IHCNuSC as a representative benchmark for CD47-stained esophageal cancer. Unlike existing publicly accessible resources that typically provide limited label granularity (detection only or binary typing), CD47-IHCNuSC addresses these limitations by targeting esophageal tissue and the CD47 biomarker under realistic IHC variability.

## Dataset Structure

```
CD47_IHCNuSC/
├── images/
│   ├── train/                # 70 images (.tif)
│   └── val/                  # 16 images (.tif)
│
├── source_annotations/
│   ├── train/                # 70 annotation files (.mat)
│   └── val/                  # 16 annotation files (.mat)
│
├── type_info_cd47_nuclei.json   # Unified nuclear subtype dictionary
└── readme_en.md                  # Dataset documentation
```

## Dataset Statistics

### Data Volume
- **Training Set**: 70 image files + 70 annotation files
- **Test/Validation Set**: 16 image files + 16 annotation files
- **Total**: 86 image-annotation pairs
- **Total Dataset Size**: ~518 MB

### Image Format
- **File Format**: TIF (Tagged Image File Format), OME-TIFF compatible
- **Image Dimensions**: 1024 × 1024 pixels
- **Color Mode**: RGB (3 channels)
- **Data Type**: uint8
- **Pixel Value Range**: [0, 255]
- **Average File Size**: ~3.1 MB per image
- **Spatial Resolution**: 0.25 μm/pixel (40× magnification)

### Annotation Format
- **File Format**: MATLAB (.mat) files
- **Annotation Content**: Instance-level segmentation masks and type-level classification labels

## Nuclear Statistics

The final dataset comprises **18,483** quality-controlled nuclei assigned to **seven** histologically and biomarker-informed subtypes spanning CD47-positive/negative tumor, immune, stromal, and rare/other categories.

### Category Distribution

| Nuclear Category | Abbrev. | Train | Test | Total | Proportion |
|-----------------|---------|-------|------|-------|------------|
| Positive-Tumor | pTu | 5,958 | 1,689 | 7,647 | 40.5% |
| Positive-Immune | pIm | 4,712 | 882 | 5,594 | 29.6% |
| Positive-Others | pOth | 17 | 8 | 25 | 0.13% |
| Negative-Tumor | nTu | 2,631 | 432 | 3,063 | 16.2% |
| Negative-Immune | nIm | 1,473 | 214 | 1,687 | 8.9% |
| Negative-Stroma | nSt | 410 | 54 | 464 | 2.5% |
| Negative-Others | nOth | 3 | 0 | 3 | 0.02% |
| **Total Nuclei** | -- | 15,204 | 3,279 | 18,483 | 100% |
| **Number of Images** | -- | 70 | 16 | 86 | -- |

**Note**: The distribution exhibits a pronounced long tail: positive-tumor and positive-immune nuclei dominate, whereas rare categories (e.g., negative-immune and stroma) are sparsely represented.

## Annotation File Structure

Each `.mat` annotation file contains the following fields:

| Field Name | Data Type | Shape | Description |
|------------|-----------|-------|-------------|
| `inst_map` | uint16/uint32 | (1024, 1024) | Instance segmentation map where each nuclear instance has a unique ID, background is 0 |
| `inst_uid` | array | (1, N) | List of instance unique identifiers, where N is the number of nuclear instances in the image |
| `inst_type` | array | (1, N) | Cell type label for each instance (corresponds to category ID in type_info) |
| `inst_centroid` | array | (N, 2) | Centroid coordinates [x, y] for each instance |
| `type_map` | uint8 | (1024, 1024) | Type classification map where each pixel location is annotated with the corresponding cell type |

### Annotation Example
Example from a validation set sample:
- **Image**: `180027-20-25-20230704151217-x115095-y82396.tif`
- **Annotation**: `180027-20-25-20230704151217-x115095-y82396.mat`
- **Number of Instances**: 411 nuclei
- **Contained Types**: Types 2, 4, 5, 7 (corresponding to positive_immune, negative_tumor, negative_immune, negative_stroma)

## Nuclear Type Definitions

The dataset defines 7 cell types (excluding background), with type information defined in `type_info_cd47_nuclei.json`:

| Type ID | Type Name | RGB Color | Description |
|---------|-----------|-----------|-------------|
| 0 | background | [1, 1, 1] | Background regions |
| 1 | positive_tumor | [255, 1, 1] | CD47-positive tumor cells |
| 2 | positive_immune | [247, 173, 242] | CD47-positive immune cells |
| 3 | positive_others | [171, 72, 247] | CD47-positive other cells |
| 4 | negative_tumor | [1, 255, 255] | CD47-negative tumor cells |
| 5 | negative_immune | [1, 255, 1] | CD47-negative immune cells |
| 6 | negative_others | [230, 179, 77] | CD47-negative other cells |
| 7 | negative_stroma | [1, 128, 1] | CD47-negative stromal cells |

**Note**: The seven main nuclear subtypes for evaluation are types 1-7 (excluding background).

### Type Information File
Type definition file path: `type_info_cd47_nuclei.json`

## Dataset Construction and Annotation Pipeline

### Image Acquisition
All CD47-immunostained esophageal whole-slide images (WSIs) were processed using a diaminobenzidine (DAB) chromogen with hematoxylin counterstain and digitized at 0.25 μm/pixel (40× magnification) on an Aperio AT2 scanner. A board-certified pathologist delineated tumor-enriched, morphologically diverse regions; patches with low tissue content were excluded. We sampled **86** non-overlapping 1024×1024 patches. To preserve real-world appearance, no stain normalization was applied.

### Annotation Process
To generate instance-level and type-level annotations, a semi-automated human-in-the-loop pipeline was adopted:

1. **Initial Segmentation**: Coarse nuclear masks were initially generated using [HoverFast](https://github.com/), a publicly available segmentation model that includes a variant pretrained for nuclear semantic segmentation on IHC-stained images.

2. **Manual Refinement**: These preliminary masks were refined through multiple rounds of manual correction in QuPath through split/merge operations and boundary correction.

3. **Type Assignment**: Trained scientists assigned preliminary subtypes; board-certified pathologists adjudicated ambiguities and harmonized criteria.

4. **Quality Control**: Final annotations underwent expert review to ensure accuracy and consistency.

### Data Format
Final annotations are released in standardized formats—OME-TIFF for images and .mat for instance masks—to ensure reproducibility and compatibility with downstream pipelines.

## File Naming Convention

Image and annotation files follow a consistent naming convention:

```
{Sample ID}-x{Coordinate X start}-y{Coordinate Y start}-x{Coordinate X end}-y{Coordinate Y end}.tif/.mat
```

Example filenames:
- `180027-20-25-20230704151217-x115095-y82396.tif`
- `180027-20-27-cd47-20221215102849_x37393_y124594_x3972_y6687.tif`
- `192257-23-20230112111644_x10832_y47096_x5129_y11451.tif`

The coordinate information in filenames indicates the position of the image within the original WSI (whole-slide image).

## Usage Instructions

### 1. Loading Images
```python
from PIL import Image
import numpy as np

# Load image
img_path = "CD47NuSC/CD47_nuclei_images/train/xxx.tif"
image = Image.open(img_path)
image_array = np.array(image)  # shape: (1024, 1024, 3), dtype: uint8
```

### 2. Loading Annotations
```python
from scipy import io

# Load annotation file
ann_path = "CD47NuSC/source_annotations/train/xxx.mat"
mat_data = io.loadmat(ann_path)

# Extract fields
inst_map = mat_data['inst_map']      # Instance segmentation map
inst_uid = mat_data['inst_uid']      # Instance ID list
inst_type = mat_data['inst_type']    # Instance types
inst_centroid = mat_data['inst_centroid']  # Instance centroids
type_map = mat_data['type_map']      # Type classification map
```

### 3. Loading Type Information
```python
import json

# Load type definitions
type_info_path = "type_info_cd47_nuclei.json"
with open(type_info_path, 'r') as f:
    type_info = json.load(f)

# type_info format: {"0": ["background", [1, 1, 1]], ...}
```

## Dataset Characteristics and Challenges

Beyond class imbalance, the benchmark captures three key challenges:

1. **Stain Heterogeneity**: Diaminobenzidine (DAB) intensity and counterstain variability across different tissue regions and WSIs.

2. **Nuclear Crowding and Adhesion**: Dense nuclear packing and overlapping instances, particularly in tumor regions, making instance separation challenging.

3. **Shape and Size Diversity**: Significant variation in nuclear morphology and size across different tissue contexts (tumor, immune, stromal regions).

Collectively, these factors stress a model's ability to jointly delineate instances and infer biomarker-sensitive types under real-world clinical heterogeneity.

## Dataset Features

1. **High Resolution**: 1024×1024 pixel images provide sufficient detail for precise nuclear segmentation.

2. **Fine-Grained Multi-Class Annotation**: Contains seven histologically and biomarker-informed nuclear subtypes (see Nuclear Type Definitions), supporting fine-grained cell classification tasks.

3. **Instance-Level Annotation**: Provides instance segmentation masks and type labels for each nucleus.

4. **Centroid Information**: Includes centroid coordinates for each instance, facilitating downstream analysis.

5. **CD47 IHC Staining**: Specifically designed for CD47 protein expression analysis in immunohistochemistry-stained images.

6. **Esophageal Cancer Focus**: Complements existing breast-centric datasets by broadening biomarker and organ coverage to esophageal IHC.

7. **Expert-Validated**: All annotations underwent review by board-certified pathologists.

8. **Real-World Variability**: Preserves natural stain appearance without normalization, reflecting clinical heterogeneity.

## Applications

This dataset is suitable for the following tasks:
- Nuclear instance segmentation
- Cell type classification
- CD47 expression status analysis (positive/negative)
- Tumor microenvironment analysis
- Immune cell identification and counting
- Multi-task learning (segmentation + classification)
- Biomarker-aware nuclear segmentation and classification (NuSC)
- Evaluation of models under clinically realistic IHC conditions

## Dataset Significance

CD47-IHCNuSC complements current breast-centric datasets and multiplex TMA corpora by broadening biomarker and organ coverage to esophageal IHC with fine-grained nuclear subtyping. The combination of biomarker specificity, fine-grained nuclear categorization, real-world stain variability, and expert-reviewed annotations makes CD47-IHCNuSC a challenging and representative benchmark. It provides a valuable resource for training and evaluating segmentation models under clinically realistic IHC conditions.

## Important Notes

1. **File Correspondence**: Image files and annotation files must correspond one-to-one; filenames (excluding extensions) should be identical.

2. **Instance IDs**: Instance IDs in `inst_map` start from 1 (0 represents background) and correspond to IDs in `inst_uid`.

3. **Type Mapping**: Type IDs in `inst_type` correspond to category IDs defined in `type_info_cd47_nuclei.json`.

4. **Tissue Type**: This dataset is derived from esophageal cancer tissue, specifically CD47-stained IHC images.

5. **No Stain Normalization**: Images preserve real-world appearance without stain normalization to maintain clinical realism.

## Citation

If you use this dataset in your research, please cite the corresponding publication:

```bibtex
@article{qin2025cd47ihcnusc,
  title        = {CD47-IHCNuSC: A Benchmark Dataset for Fine-Grained Nuclear Segmentation and Classification in CD47-Stained Esophageal Cancer},
  author       = {Qin, Lulu and Pei, Zhigang and He, Xudong and Zhou, Jiarui and Xu, Xianhong and Zhu, Zexuan},
  year         = {2025},
  note         = {Dataset available at GigaDB upon publication}
}
```


