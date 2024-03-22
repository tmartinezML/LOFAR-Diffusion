import shutil
from pathlib import Path

src = Path("/storage/tmartinez")
dst = Path("/hs/fs08/data/group-brueggen/tmartinez")

folders = [
    "image_data",
    "model_results",
    "analysis_results",
    "classifier"
]

for f in folders:
    print(f"Copying {f}.")
    shutil.copytree(src / f, dst / f)
    print(f"Copied {f}.")
print("Done.")