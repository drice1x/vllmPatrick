DEFAULT_MONITOR_CONFIG = {
    "enabled": True,
    "model_family": "qwen3_8b",
    "layers": [33, 34, 35, 36],
    "sae_dir": "/home/patrick/hacking_monitoringpipeline/saes_h8000_qwen",
    "pca_dir": "/home/patrick/hacking_monitoringpipeline/models_qwen",
    "layer_agg": "mean",
    "threshold": 0.5,
    "n_bins": 64,
    "dtype": "float16",
}