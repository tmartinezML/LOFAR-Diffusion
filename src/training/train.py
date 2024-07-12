import wandb

import utils.paths as paths
from model.config import modelConfig
from training.trainer import DiffusionTrainer
from data.datasets import TrainDataset, TrainDatasetFIRST
from utils.device_utils import set_visible_devices


if __name__ == "__main__":
    # Limit visible GPUs
    set_visible_devices(1)

    # Hyperparameters
    conf = modelConfig.from_preset("FIRST_Model")
    conf.model_name = f"FIRST_Test"

    dataset = TrainDatasetFIRST()

    trainer = DiffusionTrainer(
        config=conf,
        dataset=dataset,
    )

    wandb.init(
        project="Diffusion",
        config=conf.param_dict,
        dir=paths.ANALYSIS_PARENT / "wandb",
        # Use this for pickup:
        # id="wdh8djaz",
        # resume="must",
    )
    trainer.training_loop()
