# Validation

The release is checked with:

```bash
python -m pytest -q
python -m compileall -q configs main network tests
```

The tests cover the DCT/IDCT round trip, DCT high-pass filtering, WFE, MSCE,
PFAE, the complete output contract, loss/backward propagation, frozen
foundation backbones, optimizer parameter groups, and watershed
post-processing.

A separate GPU smoke test is provided for users who have the real SAM ViT-B
and UNI ViT-L checkpoints:

```bash
python -m tests.smoke_real \
  --sam-checkpoint /path/to/sam_vit_b_01ec64.pth \
  --uni-checkpoint /path/to/uni/pytorch_model.bin \
  --device cuda \
  --backward
```

The reference implementation was validated with real SAM and UNI checkpoints
for model construction, forward shape and finiteness, backward propagation,
and the backbone-freezing contract. Reproducing the manuscript's quantitative
test-set results additionally requires the released trained FPNuNet checkpoint
and the CD47-IHCNuSC evaluation data.

Reference audit:

| Module | Parameters |
|---|---:|
| WFE | 952 |
| MSCE | 7,384 |
| PFAE global fusion | 2,641,529 |
| PFAE skip necks | 6,336 |
| Binary decoder | 578,628 |
| HV decoder | 763,718 |
| Type decoder (five output channels) | 1,103,823 |
| Complete model | 394,898,226 |
| Trainable parameters | 5,403,954 |

The GPU backward smoke test produced a finite loss. SAM and UNI backbone
gradients were absent, while prompt-generator, fusion-neck, and decoder
gradients were present. All five output entries had the documented shapes and
finite values.
