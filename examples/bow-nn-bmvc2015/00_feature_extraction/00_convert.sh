#!/bin/bash

mkdir -p caches
mkdir -p log

# list of all dense trajectory files in ../data/dense_traj
TRAJ_FILE_LIST=$(ls --color=none ../data/dense_traj/*.gz)

# loop over all those files
for TRAJ_FILE in $TRAJ_FILE_LIST; do

    # extract the video name (remove path and .gz from string)
    VIDEO_NAME=$(echo $TRAJ_FILE | sed -e "s#.*/##" | sed -e "s#\.gz##")

    ### CONVERT DENSE TRAJECTORIES TO SQUIRREL CACHE ###########################
    OPTIONS="--log-file=log/convert.$VIDEO_NAME.log \
             --action=external-to-cache \
             --external-type=dense-trajectories \
             --dense-trajectory-conversion.trajectory-file=${TRAJ_FILE} \
             --dense-trajectory-conversion.feature-cache-basename=caches/${VIDEO_NAME}"
    ../../../executables/converter $OPTIONS
    ############################################################################
done
