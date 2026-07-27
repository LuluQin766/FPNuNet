"""Paper optimizer and step-level learning-rate schedule."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from configs import CONFIG


def build_optimizer(
    model: nn.Module,
    config: Mapping = CONFIG,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.LambdaLR]:
    """Create the manuscript AdamW parameter groups and warmup/decay schedule."""
    optimizer_config = config["optimizer"]
    prompt_parameters = [
        parameter
        for encoder in (model.sam_encoder, model.uni_encoder)
        for parameter in encoder.prompt_generator.parameters()
        if parameter.requires_grad
    ]
    prompt_ids = {id(parameter) for parameter in prompt_parameters}
    component_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in prompt_ids
    ]
    if not prompt_parameters or not component_parameters:
        raise ValueError("Both prompt and component parameter groups must be non-empty")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": prompt_parameters,
                "lr": optimizer_config["prompt_lr"],
                "name": "prompt_generators",
            },
            {
                "params": component_parameters,
                "lr": optimizer_config["component_lr"],
                "name": "trainable_components",
            },
        ],
        weight_decay=optimizer_config["weight_decay"],
    )

    warmup_steps = int(optimizer_config["warmup_steps"])
    decay_steps = tuple(int(step) for step in optimizer_config["decay_steps"])
    decay_factor = float(optimizer_config["decay_factor"])

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(step + 1, 1) / warmup_steps
        completed_decays = sum(step >= boundary for boundary in decay_steps)
        return decay_factor**completed_decays

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    return optimizer, scheduler
