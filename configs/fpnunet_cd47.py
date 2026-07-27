"""Canonical configuration matching the FPNuNet manuscript."""

CONFIG = {
    "input_size": [128, 128],
    "classes": ["background", "pTu", "pIm", "nTu", "nOth"],
    "loss": {
        "binary": {
            "bce": 1.0,
            "dice": 1.0,
            "focal": 1.0,
            "boundary": 0.5,
        },
        "hv": {"mse": 1.0, "msge": 1.0, "foreground_only": True},
        "type": {
            "cross_entropy": 1.0,
            "dice": 1.0,
            "focal": 1.0,
            "iou": 1.0,
            "class_weights": [0.25, 1.0, 1.0, 1.0, 1.0],
            "focal_gamma": 2.0,
        },
    },
    "optimizer": {
        "name": "AdamW",
        "prompt_lr": 1e-3,
        "component_lr": 5e-4,
        "weight_decay": 1e-2,
        "warmup_steps": 1000,
        "decay_steps": [20000, 27000],
        "decay_factor": 0.5,
        "max_steps": 30000,
        "precision": "bf16-mixed",
    },
}
