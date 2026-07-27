"""Minimal patch inference CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from network import build_model
from main.postprocess import logits_to_instances


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--uni-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB").resize((128, 128))
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    model = build_model(
        args.sam_checkpoint, args.uni_checkpoint, args.checkpoint, args.device
    ).eval()
    with torch.inference_mode():
        output = model(tensor.unsqueeze(0).to(args.device))
    instances, nuclear_types = logits_to_instances(
        output["bin"][0], output["hv"][0], output["tp"][0]
    )
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / "instance_map.npy", instances)
    np.save(output_path / "type_map.npy", nuclear_types)


if __name__ == "__main__":
    main()
