import pandas as pd
import numpy as np
from scipy.stats import rv_histogram

from model.sampler import Sampler
from utils.device_utils import set_visible_devices
from model.model_utils import load_data_transforms
import utils.logging
import utils.paths as paths


logger = utils.logging.get_logger(__name__)


if __name__ == "__main__":

    # Set up devices
    n_gpu = 1
    dev_ids = set_visible_devices(n_gpu)
    logger.info(f"Using GPU {dev_ids[:n_gpu]}")

    # Sampling parameters
    model_name = "Prototypes_Model_SizeCond"
    n_samples = 8_000

    # Get context function
    logger.info("Getting size context function from training data...")
    mask_metadata = pd.read_hdf(
        paths.LOFAR_SUBSETS["prototypes"],
        key="mask_metadata",
    )
    sizes = mask_metadata["feret_diameter_max"]
    sizes = sizes[mask_metadata["Model_Radius"] <= 40]
    size_transform = load_data_transforms(model_name)["mask_sizes"]
    model_size_distribution = np.histogram(sizes, bins=100, density=True)
    size_rvs = rv_histogram(model_size_distribution)
    size_context = size_rvs.rvs(size=n_samples)
    size_context_tr = size_transform.transform(size_context.reshape(-1, 1))

    # Initialize sampler
    sampler = Sampler(n_samples=n_samples, n_devices=n_gpu)

    # Sample from model
    logger.info("Sampling with context...")
    sampler.sample(
        model_name,
        #
        # Use this when sampling from LOFAR model:
        context=size_context_tr,
        #
        # Use this when sampling from FIRST model:
        # labels=sampler.get_labels(),
        #
        comment="Size-Conditioned",
    )

    logger.info("Sampling without context...")
    sampler.sample(
        model_name,
        #
        # Use this when sampling from LOFAR model:
        # context_fn=sampler.get_fpeak_model_dist(paths.LOFAR_SUBSETS["0-clip"]),
        #
        # Use this when sampling from FIRST model:
        # labels=sampler.get_labels(),
        #
        comment="Unconditioned",
    )
