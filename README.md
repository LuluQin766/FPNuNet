# FPNuNet

Official compact implementation of **FPNuNet: A Frequency-Aware
Prompt-Guided Network for Nuclear Segmentation and Classification in
Immunohistochemistry Images**.

This release contains the final model, loss, optimizer schedule, patch
inference, and HoVer-Net-style instance post-processing. Historical models,
ablations, experiment launchers, datasets, logs, and pretrained weights are
not included.

## Architecture

FPNuNet processes RGB patches of shape `B x 3 x 128 x 128` through four
parallel encoders:

- a frozen SAM ViT-B structural backbone with a trainable `8 x 8` patch
  projection and a DCT high-pass prompt generator;
- a frozen UNI ViT-L semantic backbone with prompts injected into alternating
  transformer layers and features collected from layers 3, 12, and 21;
- a Haar wavelet feature encoder (WFE) producing LL/LH/HL/HH descriptors;
- a three-scale spatial context encoder (MSCE).

A four-way DCT-enabled PFAE neck fuses SAM, UNI, WFE, and MSCE features.
Two DCT-PFAE necks enhance the high-resolution UNI skips used by the decoder.
The binary decoder runs first and supplies hierarchical guidance to the HV and
type branches. The HV branch uses bilinear upsampling, skip cross-attention,
dense refinement, and binary gating. In the type branch, learned prompts are
the attention Query and pooled decoder features are Key/Value.

The model returns raw logits or raw regression values; it does not apply
`sigmoid` or `softmax` internally.

With SAM ViT-B and UNI ViT-L loaded, the five-class release contains
`394,898,226` total parameters and `5,403,954` trainable parameters (1.37%),
consistent with the manuscript's rounded `394.8M / 5.4M` report.

## Installation

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the SAM ViT-B checkpoint `sam_vit_b_01ec64.pth` from Meta's Segment
Anything release and the UNI ViT-L `pytorch_model.bin` checkpoint from the
official UNI release. Their code and weights remain subject to their own
licenses.

## Model construction

```python
import torch
from network import build_model

model = build_model(
    sam_checkpoint="/path/to/sam_vit_b_01ec64.pth",
    uni_checkpoint="/path/to/uni/pytorch_model.bin",
    fpnunet_checkpoint="/path/to/fpnunet.pt",  # optional
    device="cuda",
).eval()

# Input is RGB, float32, scaled to [0, 1].
image = torch.rand(2, 3, 128, 128, device="cuda")
with torch.inference_mode():
    output = model(image)
```

Output contract:

| Key | Shape | Meaning |
|---|---|---|
| `bin` | `[B, 1, 128, 128]` | Binary-nucleus logits |
| `boundary` | `[B, 1, 128, 128]` | Boundary logits |
| `hv` | `[B, 2, 128, 128]` | Horizontal/vertical regression |
| `tp` | `[B, 5, 128, 128]` | Type logits |
| `type_aux` | two lower-resolution tensors | Optional deep-supervision logits |

The type-channel order is `background`, `pTu`, `pIm`, `nTu`, `nOth`.
`build_model()` accepts a plain state dictionary or a checkpoint containing
`state_dict`, `model_state_dict`, or `model`; common training prefixes are
removed automatically.

## Loss and optimizer

```python
from main import FPNuNetLoss, build_optimizer

criterion = FPNuNetLoss()
optimizer, scheduler = build_optimizer(model)
```

The paper configuration is in `configs/fpnunet_cd47.py`:

- binary BCE + Dice + focal, with `0.5 x` boundary loss;
- foreground-normalized HV MSE + Sobel MSGE;
- equally weighted type CE + Dice + focal + soft IoU;
- CE weights `[0.25, 1.0, 1.0, 1.0, 1.0]`;
- AdamW with prompt-generator LR `1e-3`, other trainable-component LR `5e-4`,
  1,000-step warmup, `0.5 x` decay at steps 20,000 and 27,000, and 30,000
  total steps with bf16 mixed precision.

## Patch inference

```bash
python -m main.infer \
  --image sample.png \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --uni-checkpoint /path/to/uni/pytorch_model.bin \
  --checkpoint /path/to/fpnunet.pt \
  --output outputs \
  --device cuda
```

The command writes `instance_map.npy` and `type_map.npy`. It is a patch-level
reference implementation and does not resize predictions back to the source
image. Prepare `128 x 128` patches when spatial correspondence must be
preserved.

## Validation

```bash
pytest -q
```

See `VALIDATION.md` for the real-checkpoint GPU smoke command and the exact
validation boundary.

## Citation

```bibtex
@article{qin2026fpnunet,
  title   = {FPNuNet: A Frequency-Aware Prompt-Guided Network for Nuclear
             Segmentation and Classification in Immunohistochemistry Images},
  author  = {Qin, Lulu and Pei, Zhigang and He, Xudong and Zhou, Jiarui and
             Xu, Xianhong and Zhu, Zexuan},
  journal = {GigaScience},
  year    = {2026}
}
```

Please also cite the original SAM and UNI publications.

## License

The FPNuNet source code is released under the MIT License. SAM and UNI code
and checkpoints are not redistributed by this repository.
