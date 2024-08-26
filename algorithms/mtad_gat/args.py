# import argparse

hyper_parameter = {
    # Preprocess
    "least_train_days": 20,
    "least_nan_percent": 0.1,
    # Data Param
    "lookback": 60,
    "normalize": True,
    "spec_res": False,
    # --- Model Param ---
    # 1D conv layer
    "kernel_size": 7,
    # GAT layers
    "use_gatv2": True,
    "feat_gat_embed_dim": None,
    "time_gat_embed_dim": None,
    # GRU layer
    "gru_n_layers": 2,
    "gru_hid_dim": 150,
    # Forecasting Model
    "fc_n_layers": 3,
    "fc_hid_dim": 150,
    # Reconstruction Model
    "recon_n_layers": 1,
    "recon_hid_dim": 150,
    # Others
    "alpha": 0.2,
    # --- Train params ---
    "p_n_epochs": 30,
    "val_split": 0.1,
    "p_bs": 512,
    "p_init_lr": 0.0005,
    "shuffle_dataset": True,
    "dropout": 0.3,
    "use_cuda": True,
    "print_every": 1,
    "log_tensorboard": True,

    # --- Predictor params ---
    "scale_scores": False,
    "use_mov_av": False,
    "gamma": 1,
    "level": 0.9900,
    "q": 0.005,
    "reg_level": 1,
    "dynamic_pot": False,

    # Serving params
    "forecast_len": 30,
}