#!/bin/bash

# To be run within singularity container

# Activate conda environment
cd /tmartinez
source envs/losito_venv/bin/activate

# Run losito with desired parset
cd /tmartinez/sky_maps/test_map_80/losito
losito losito.parset
