config = {
    "ARTIFACTS_DIR": "../artifacts",
    "MODEL": {
        "img_size": [224, 224, 3],
        "img_encoder": {"pretrained": True},
        "text_encoder": {"context_len": 32, "embed_size": 64, "n_heads": 4, "n_layers": 4, "dropout": 0.1},
    },
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
