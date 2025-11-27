from box import Box

name = "CD47_nuclei"
dataset_root = f"/root/{name}_patches_v5"

config = {
    "devices": None,
    "batch_size": 12,
    "accumulate_grad_batches": 1,
    "num_workers": 8,
    "out_dir": "/root/FPNuNet_results",
    "log_dir": None,
    "log_train_images_path": None,
    "log_val_images_path": None,
    "opt": {
        "num_epochs": 80,
        "learning_rate": 5e-4,
        "weight_decay": 1e-2, #1e-2,
        "precision": 32, # "16-mixed" 
        "steps":  [72 * 25, 72 * 29],
        "warmup_steps": 72,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/root/pretrained_models/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
        "prompt_dim": 256,
        "prompt_decoder": False,
        "dense_prompt_decoder": True,

        "extra_encoder": 'uni_v1',
        "extra_type": "multihead_v15",
        "extra_checkpoint": "/root/pretrained_models/uni/pytorch_model.bin",
    },
    "loss": {
        "bin": {
            "focal_cof": 1.0,
            "bce_cof": 1.0,
            "dice_cof": 1.0,
            "boun_loss_cof": 0.5,
            "aux_loss_cof": 0.25
        },
        "tp": {
            "focal_cof": 1.0,
            "dice_cof": 1.0,
            "ce_cof": 1.0,
            "iou_cof": 1.0,
            "aux_loss_cof": 0.25,
            "class_weights": [
                0.0,    # background
                1.0,    # positive_tumor
                0.5,    # positive_immune_related
                0.25,    # positive_others
                1.0,    # negative_tumor
                0.25,    # negative_immune_related
                0.0,    # negative_others
                0.25,    # negative_stroma
                0.0     # ambiguous
            ],
        },
        "hv": {
            "mse_cof": 1.0,
            "msge_cof": 1.0,
            "aux_loss_cof": 0.25
        }

    },
    "dataset": {
        "train_h5_file_path": f"{dataset_root}/gt_{name}_train_128x128_64x64_img_inst_type_hv.h5",
        "test_h5_file_path": f"{dataset_root}/gt_{name}_valid_128x128_64x64_img_inst_type_hv.h5",
        "num_classes": 8,
        "type_info_path": "../CD47_IHCNUSC/type_info_cd47_nuclei.json",
        "color_dict": {
            "0" : ["background", [1,   1, 1]],
            "1" : ["positive_tumor", [255, 1, 1]],
            "2" : ["positive_immune_related", [247, 173, 242]],
            "3" : ["positive_others", [171, 72, 247]],
            "4" : ["negative_tumor", [1, 255, 255]],
            "5" : ["negative_immune_related", [1, 255, 1]],
            "6" : ["negative_others", [230, 179, 77]],
            "7" : ["negative_stroma", [1, 128, 1]],
            "8" : ["ambiguous", [1, 1, 255]]
        },
        "class_weights": [
            0.0,    # background
            1.0,    # positive_tumor
            0.5,    # positive_immune_related
            0.5,    # positive_others
            1.0,    # negative_tumor
            0.5,    # negative_immune_related
            0.5,    # negative_others
            0.0,    # negative_stroma
            0.0     # ambiguous
        ],

        "ignored_classes": (0),
        "ignored_classes_metric": None, # if we do not count background, set to 0 (bg class)
        "image_hw": (128, 128), # default is 1024, 1024

        "feature_input": False, # or "True" for *.pt features
        "dataset_mean": (1, 1, 1),
        "dataset_std": (1, 1, 1),
    }
}

cfg = Box(config)