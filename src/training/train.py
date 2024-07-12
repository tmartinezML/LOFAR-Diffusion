import utils.paths as paths
import data.datasets as datasets
import utils.device_utils as device_utils
from model.config import ModelConfig
from training.trainer import DiffusionTrainer


if __name__ == "__main__":
    # Limit visible GPUs
    device_utils.set_visible_devices(1)

    # Set model preset:
    # (i.e. name of the json file in the model_configs directory)
    model_preset = ""

    # Hyperparameters
    conf = ModelConfig.from_preset(model_preset)

    # Change the name if you want:
    # (otherwise default name is used)
    # conf.model_name = "Alternative_Name"

    # Load dataset:
    dataset = datasets.TrainDatasetFIRST()

    # Initialize trainer
    trainer = DiffusionTrainer(config=conf, dataset=dataset)

    # Start training
    trainer.training_loop()
