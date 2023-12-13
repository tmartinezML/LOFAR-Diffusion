from utils.config_utils import modelConfig

def InitModel_EDM_config():
    conf = modelConfig(
        # Unet parameters
        model_name = "InitModel_EDM_lr=2e-5_bsize=256_(1,2,2,2)",
        model_type = "EDMPrecond",
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 2, 2),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        dropout = 0.0,
        pretrained_model = None,
        # Diffusion parameters
        timesteps = 1000,
        learn_variance = False,
        # Training parameters
        batch_size = 256,
        iterations = 200_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 500,
        snapshot_interval = 20_000,
        val_every = 2500,
        validate_ema = True,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 2,
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
        init_channels = 128,
        channel_mults = (1, 2, 2),
        norm_groups = 32,
        attention_levels = 2,
        attention_heads = 2,
        attention_head_channels = 8,
        dropout = 0.0,
        pretrained_model = None,
        # Diffusion parameters
        timesteps = 1000,
        learn_variance = False,
        # Training parameters
        batch_size = 1,
        iterations = 10_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 10,
        snapshot_interval = 20,
        val_every = 10,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 1,
    )
    return conf

def EDM_config():
    conf = modelConfig(
        # Unet parameters
        model_name = "EDM",
        model_type = "EDMPrecond",
        image_size = 80,
        image_channels = 1,
        init_channels = 160,
        channel_mults = (1, 2, 2, 2),
        norm_groups = 32,
        attention_levels = 3,
        attention_heads = 4,
        attention_head_channels = 32,
        dropout = 0.0,
        pretrained_model = None,
        # Diffusion parameters
        timesteps = 1000,
        learn_variance = False,
        # Training parameters
        batch_size = 256,
        iterations = 200_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 500,
        snapshot_interval = 20_000,
        val_every = 2500,
        validate_ema = True,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 2,
    )
    return conf


def EDM_small_config():
    conf = modelConfig(
        # Unet parameters
        model_name = "EDM_small",
        model_type = "EDMPrecond",
        image_size = 80,
        image_channels = 1,
        init_channels = 128,
        channel_mults = (1, 2, 2),
        norm_groups = 32,
        attention_levels = 2,
        attention_heads = 2,
        attention_head_channels = 32,
        dropout = 0.0,
        pretrained_model = None,
        # Diffusion parameters
        timesteps = 1000,
        # Training parameters
        batch_size = 256,
        iterations = 200_000,
        learning_rate = 2e-5,
        ema_rate = 0.9999,
        log_interval = 500,
        snapshot_interval = 25_000,
        val_every = 2500,
        validate_ema = True,
        write_output = True,
        override_files = True,
        # Parallel training
        n_devices = 2,
    )
    return conf