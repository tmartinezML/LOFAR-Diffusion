from model.sampler import Sampler
from utils.device_utils import set_visible_devices
import utils.logging

logger = utils.logging.get_logger(__name__)


if __name__ == "__main__":

    # Set up devices
    n_gpu = 2
    dev_ids = set_visible_devices(n_gpu)
    logger.info(f"Using GPU {dev_ids[:n_gpu]}")

    # Sampling parameters
    model_name = "Prototypes_Model"
    n_samples = 8_000

    # Initialize sampler
    sampler = Sampler(n_samples=n_samples, n_devices=n_gpu)

    # Sample from model
    sampler.sample(
        model_name,
        #
        # Use this when sampling from LOFAR model:
        # context_fn=sampler.get_fpeak_model_dist(paths.LOFAR_SUBSETS["0-clip"]),
        #
        # Use this when sampling from FIRST model:
        # labels=sampler.get_labels(),
        #
    )
    logger.info(f"Finished sampling {n_samples:_} samples from {model_name}.")
