"""GPU smoke test using real SAM and UNI checkpoints."""

from __future__ import annotations

import argparse
import json

import torch

from main import FPNuNetLoss
from network import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--uni-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    model = build_model(args.sam_checkpoint, args.uni_checkpoint, device=args.device).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 128, 128, device=args.device))
    backward_report = None
    if args.backward:
        model.train()
        image = torch.randn(1, 3, 128, 128, device=args.device)
        prediction = model(image)
        target = {
            "bin": torch.randint(0, 2, (1, 1, 128, 128), device=args.device).float(),
            "boundary": torch.randint(0, 2, (1, 1, 128, 128), device=args.device).float(),
            "hv": torch.randn(1, 2, 128, 128, device=args.device),
            "tp": torch.randint(0, 5, (1, 128, 128), device=args.device),
        }
        loss = FPNuNetLoss().to(args.device)(prediction, target)["total"]
        loss.backward()
        backward_report = {
            "loss": float(loss.detach().cpu()),
            "finite": bool(torch.isfinite(loss).item()),
            "sam_backbone_grad": any(
                parameter.grad is not None
                for parameter in model.sam_encoder.image_encoder.blocks.parameters()
            ),
            "uni_backbone_grad": any(
                parameter.grad is not None
                for parameter in model.uni_encoder.backbone.blocks.parameters()
            ),
            "sam_prompt_grad": any(
                parameter.grad is not None
                for parameter in model.sam_encoder.prompt_generator.parameters()
            ),
            "fusion_grad": any(
                parameter.grad is not None for parameter in model.fusion_neck.parameters()
            ),
            "decoder_grad": any(
                parameter.grad is not None for parameter in model.type_decoder.parameters()
            ),
        }
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(
        json.dumps(
            {
                "torch": torch.__version__,
                "device": str(args.device),
                "total_parameters": total,
                "trainable_parameters": trainable,
                "modules": model.parameter_summary(),
                "backward": backward_report,
                "outputs": {
                    key: [list(item.shape) for item in value]
                    if isinstance(value, list)
                    else list(value.shape)
                    for key, value in output.items()
                },
                "finite": {
                    key: all(torch.isfinite(item).all().item() for item in value)
                    if isinstance(value, list)
                    else torch.isfinite(value).all().item()
                    for key, value in output.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
