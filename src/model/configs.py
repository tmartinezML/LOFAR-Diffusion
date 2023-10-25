from utils.config_utils import modelConfig

def OptiModel_Initial_config():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "OptiModel_Initial",
        use_improved_unet = False,
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        # Diffusion parameters
        timesteps = 250,
        schedule = "linear",
        learn_variance = False,
        # Training parameters
        batch_size = 128,
        iterations = 10000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 100,
        write_output = True,
        override_files = True,
        # Parallel training
        train_parallel = False,
        n_devices = 3,
    )
    return conf

def OptiModel_ImprovedUnet_config():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "OptiModel_ImprovedUnet_Default",
        model_type = "ImprovedUnet",
        use_improved_unet = True,
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        # Diffusion parameters
        timesteps = 250,
        schedule = "cosine",
        learn_variance = True,
        # Training parameters
        batch_size = 128,
        iterations = 10000,
        learning_rate = 3e-5,
        ema_rate = 0.9999,
        log_interval = 250,
        val_every = 500,
        write_output = True,
        override_files = True,
        optimizer = "Adam",
        # Parallel training
        n_devices = 3,
    )
    return conf

def InitModel_EDM_config():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "InitModel_EDM",
        model_type = "EDMPrecond",
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        dropout = 0.0,
        # Diffusion parameters
        timesteps = 1000,
        learn_variance = False,
        # Training parameters
        batch_size = 256,
        iterations = 200_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 500,
        snapshot_interval = 40_000,
        val_every = 500,
        validate_ema = True,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 3,
    )
    return conf

def DummyConfig():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name = "Dummy",
        model_type = "EDMPrecond",
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 4, 8),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        dropout = 0.1,
        # Diffusion parameters
        timesteps = 1000,
        learn_variance = False,
        # Training parameters
        batch_size = 1,
        iterations = 10_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 10,
        val_every = 10,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 1,
    )
    return conf