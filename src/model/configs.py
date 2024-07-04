from model.config_utils import modelConfig


def DummyConfig():
    # Hyperparameters
    conf = modelConfig(
        # Unet parameters
        model_name="Dummy",
        model_type="EDMPrecond",
        image_size=80,
        image_channels=1,
        init_channels=128,
        channel_mults=(1, 2, 2),
        norm_groups=32,
        attention_levels=2,
        attention_heads=2,
        attention_head_channels=8,
        dropout=0.1,
        pretrained_model=None,
        # Training parameters
        batch_size=128,
        iterations=50,
        learning_rate=2e-5,
        ema_rate=0.9999,
        log_interval=5,
        snapshot_interval=10,
        val_every=50,
        sigma_data=0.5,
        P_mean=-2.5,
        P_std=1.8,
        validate_ema=True,
        write_output=True,
        override_files=True,
        # Parallel training
        n_devices=1,
    )
    return conf


def EDM_small_config():
    conf = modelConfig(
        # Unet parameters
        model_name="EDM_small",
        model_type="EDMPrecond",
        image_size=80,
        image_channels=1,
        init_channels=128,
        channel_mults=(1, 2, 2),
        norm_groups=32,
        attention_levels=2,
        attention_heads=2,
        attention_head_channels=32,
        dropout=0.1,
        pretrained_model=None,
        # Training hyperparameters
        batch_size=256,
        iterations=100_000,
        learning_rate=2e-5,
        ema_rate=0.9999,
        sigma_data=0.5,
        P_mean=-2.5,
        P_std=1.8,
        # Training settings
        log_interval=1000,
        snapshot_interval=20_000,
        val_every=2500,
        validate_ema=True,
        write_output=True,
        override_files=False,
        # Parallel training
        n_devices=2,
    )
    return conf


def EDM2_small_config():
    conf = modelConfig(
        # Unet parameters
        model_name="EDM2",
        model_type="EDMPrecond",
        image_size=80,
        image_channels=1,
        init_channels=128,
        channel_mults=(1, 2, 2),
        attention_levels=2,
        attn_head_dim=32,
        dropout=0.0,
        pretrained_model=None,
        # Training hyperparameters
        batch_size=256,
        iterations=100_000,
        learning_rate=2e-5,
        ema_rate=0.9999,
        sigma_data=0.5,
        P_mean=-2.5,
        P_std=1.8,
        # Training settings
        log_interval=1000,
        snapshot_interval=20_000,
        val_every=2500,
        validate_ema=True,
        write_output=True,
        override_files=False,
        # Parallel training
        n_devices=2,
    )
    return conf


def FIRST_labeled_config():
    conf = modelConfig(
        # Unet parameters
        model_name="FIRST_labeled",
        model_type="EDMPrecond",
        image_size=80,
        image_channels=1,
        init_channels=128,
        channel_mults=(1, 2, 2),
        norm_groups=32,
        attention_levels=2,
        attention_heads=2,
        attention_head_channels=32,
        dropout=0.1,
        pretrained_model=None,
        label_dropout=0.1,
        n_labels=4,
        # Diffusion parameters
        timesteps=1000,
        # Training hyperparameters
        batch_size=128,
        iterations=20_000,
        learning_rate=2e-5,
        ema_rate=0.9999,
        sigma_data=0.5,
        P_mean=-2.5,
        P_std=1.8,
        # Training settings
        log_interval=1000,
        snapshot_interval=2_000,
        val_every=500,
        validate_ema=True,
        write_output=True,
        override_files=False,
        # Parallel training
        n_devices=1,
    )
    return conf
