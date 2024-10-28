#!/bin/bash

# To be run within singularity container

# Activate conda environment
cd /tmartinez
# source envs/losito_venv/bin/activate

# Run ddf-pipeline
cd /tmartinez/sky_maps/test_map_80/ddf-pipeline
pipeline.py ./config.cfg
