#!/bin/bash

# To be run within singularity container

# Activate virtual environment (this is venv, not conda)
cd /tmartinez
source envs/losito_venv/bin/activate

# Run synthms
cd /tmartinez/sky_maps/test_map_80/synthms
synthms --name test_map_80 --start 5050307679.0060005 --tobs 8 --ra 0.000545415391248228 --dec 0.40832415928049587 --station HBA --minfreq 144000000 --maxfreq 144000000 --chanpersb 2
