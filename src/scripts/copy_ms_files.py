from casacore.tables import table
import numpy as np
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import utils.paths as paths

map_name = "map_2.5deg_max80_1e-7JyLimit"
map_folder = paths.SKY_MAP_PARENT / map_name
synthms_folder = map_folder / "synthms_DP3"
synthms_folder.mkdir(exist_ok=True)

ms_list = sorted(list(synthms_folder.glob("*.MS")))


def get_mean_freq(ms):
    with table(str(ms) + "/SPECTRAL_WINDOW", ack=False) as t:
        freqs = t.getcol("CHAN_FREQ")[0] * 1e-6  # MHz
    return np.mean(freqs)


DEF_DIR = paths.BASE_PARENT / "src/maps/default_files/synthms"


def copy_ms_to_defdir(ms_dir):
    fmean = get_mean_freq(ms_dir)
    target_name = ms_dir.name.replace(
        ms_dir.stem.split("_")[-1], f"{int(np.round(fmean))}MHz"
    )
    if (DEF_DIR / target_name).exists():
        print(f"MS {target_name} already exists - deleting.")
        shutil.rmtree(DEF_DIR / target_name)
    shutil.copytree(ms_dir, DEF_DIR / target_name)


for ms in tqdm(ms_list):
    copy_ms_to_defdir(ms)
