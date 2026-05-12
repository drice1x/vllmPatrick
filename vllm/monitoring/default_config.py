DEFAULT_MONITOR_CONFIG = {
    "enabled": True,
    "model_family": "qwen3_8b",
    "layers": [28, 29, 30, 31], 
    #"layers": [32, 33, 34, 35], qwen 
    #"layers": [24, 25, 26, 27], falcon
    "sae_dir": "/home/patrick/hacking_monitoringpipeline/saes_h8000_llama",
    "pca_dir": "/home/patrick/hacking_monitoringpipeline/models_llama",
    "layer_agg": "mean",
    "threshold": 0.5,
    "n_bins": 64,
    "dtype": "float16",
}