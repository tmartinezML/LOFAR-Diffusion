#!/bin/bash

export PATH=$PATH:/hs/fs08/data/group-brueggen/tmartinez/trecs/bin
export LD_LIBRARY_PATH=/hs/fs08/data/group-brueggen/tmartinez/software/cfitsio/lib
# If the script is run from the directory that contains the parameter
# file ($1), this should work.
trecs -c -w -p $1