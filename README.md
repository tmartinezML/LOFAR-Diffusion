# Overview
This repository holds the software implementation of the paper [Simulating images of radio galaxies with diffusion models](https://arxiv.org/abs/2410.07794). We implement a diffusion model to generate images of radio galaxies. The model is trained on data from the [LoTSS-DR2](https://lofar-surveys.org/dr2_release.html). If you use this software or parts of this software for your work, please cite the publication.


# Installation

1. Clone the repository to your local machine:
```bash
git clone https://github.com/tmartinezML/LOFAR-Diffusion.git
cd LOFAR-Diffusion
```

2. If desired, create a new virtual environment. Make sure to use python 3.11. Install the required packages to your environment:
```bash
pip install -r src/requirements.txt
```
__Important__: If you use a python version >= 3.12, you will run into compatibility issues with the required packages!

3. By default, data will be stored in the repository directory. You can change this by setting the STORAGE_PARENT path in  src/utils/paths.py

4. Run src/utils/paths.py:
```bash
python -m src.utils.paths
```
This will set up the correct folder structure and download trained model weights, as well as the LOFAR training set. The FIRST dataset will be downloaded when it is first initialized.

# Usage

Training is executed with the script training/train.py. Settings are defined with a json configuration file that has to be specified inside the script. Example configs are given in model/configs. Also, the training dataset is specified in the script. Different options are defined in data/datasets.py
