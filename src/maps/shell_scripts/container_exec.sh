#!/bin/bash

# To get image:
# singularity build pill.simg docker://revoltek/pill:20240201

# Add singularity to PATH
export PATH=$PATH:/opt/singularity/bin

# Set cache dir
export SINGULARITY_CACHEDIR=/hs/fs08/data/group-brueggen/tmartinez/.singularity/cache

# Make sure to run in correct dir
cd /hs/fs08/data/group-brueggen/tmartinez

# Run script inside singularity.
# For mounting: -bind /source_on_host:/destination_in_container\
# --bind /hs/fs08/data/group-brueggen/tmartinez:/tmartinez,/hsopt/anaconda3:/hsopt/anaconda3\
singularity exec --pid --writable-tmpfs --containall --cleanenv --no-home\
 --bind /hs/fs08/data/group-brueggen/tmartinez:/tmartinez\
 --workdir /hs/fs08/data/group-brueggen/tmartinez\
 singularity_pills/losito_pill.simg bash $1
 