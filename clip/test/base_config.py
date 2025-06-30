config = {
    "ARTIFACTS_DIR": "../artifacts",
    "MODEL": {"model_name": "", "img_size": [224, 224, 3]},
    "DATA": {"dataset": "Flickr8kDataset", "batch_size": 2, "root_dir": "../data"},
    "OPTIM": {
        "loss": "",
        "optimizer": "AdamW",
        "lr": 0.0001,
        "num_epochs": 100,
        "eval_every": 5,
        "gradient_clip": True,
        "scheduler": "",
        "T_max": 100,
        "eta_min": 0.000001,
    },
}
