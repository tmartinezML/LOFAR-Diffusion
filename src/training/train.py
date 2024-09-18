import wandb

import utils.paths as paths
import data.datasets as datasets
from model.config import modelConfig
from training.trainer import DiffusionTrainer
from utils.device_utils import set_visible_devices


if __name__ == "__main__":
    # Limit visible GPUs
    set_visible_devices(2)

    # Hyperparameters
    conf = modelConfig.from_preset("LOFAR_Model")
    conf.model_name = f"Prototypes_Model_SizeCond"
    conf.iterations = 150_000
    conf.context = ['mask_sizes_tr']

    dataset = datasets.LOFARPrototypesDataset(
        paths.LOFAR_SUBSETS["prototypes"],
        img_size=conf.image_size,
        train_mode=True,
        attributes=[],
    )

    trainer = DiffusionTrainer(
        config=conf,
        dataset=dataset,
        # pickup=True,
    )

    wandb.init(
        project="Diffusion",
        config=conf.param_dict,
        dir=paths.ANALYSIS_PARENT / "wandb",
        # Use this for pickup:
        # id="mm5hsmh2",
        # resume="must",
    )
    trainer.training_loop()
