#!/bin/bash

source ../../tools/tools.sh

mkdir -p results
mkdir -p log

export OMP_NUM_THREADS=1 # the number of threads you want to use
USE_GPU="true" # set to "false" if you do not want to use a gpu

# list of all dense trajectory files in dense_traj
TRAJ_FILE_LIST=$(ls --color=none ../data/dense_traj/*.gz)

# loop over all those files
for TRAJ_FILE in $TRAJ_FILE_LIST; do

    VIDEO_NAME=$(echo $TRAJ_FILE | sed -e "s#.*/##" | sed -e "s#\.gz##")


    ### CONVERT DENSE TRAJECTORIES TO SQUIRREL CACHE ###########################
    echo -e "\n\nCONVERT DENSE TRAJECTORIES TO SQUIRREL CACHE\n\n"

    OPTIONS="--log-file=log/convert.$VIDEO_NAME.log \
             --action=external-to-cache \
             --external-type=dense-trajectories \
             --dense-trajectory-conversion.trajectory-file=${TRAJ_FILE} \
             --dense-trajectory-conversion.feature-cache-basename=results/${VIDEO_NAME}"
    run "../../../executables/converter $OPTIONS"
    
    ############################################################################


    ### COMPUTE INTEGRAL SEQUENCE OF FISHER VECTORS ############################
    echo -e "\n\nCOMPUTE INTEGRAL SEQUENCE OF FISHER VECTORS\n\n"

    # determine length of video (count number of framewise labels)
    VIDEO_LEN=$(wc -l ../data/framewise-labels/${VIDEO_NAME} | cut -f1 -d' ')

    for DESCRIPTOR in traj hog hof mbhx mbhy; do
        OPTIONS="--log-file=log/integral-sequence.${DESCRIPTOR}.${VIDEO_NAME}.log \
                 --use-gpu=$USE_GPU"
        # ensure that video has correct number of frames
        # if no trajectories are for frames at beginning or and of video, zeros are padded
        OPTIONS="--feature-quantization.ensure-sequence-length=${VIDEO_LEN} \
                 $OPTIONS"
        # specify mean and pca matrix for feature preprocessing
        OPTIONS="--vector-sub.vector=pca/mean.${DESCRIPTOR}.vector.gz \
                 --matrix-mul.matrix=pca/pca.${DESCRIPTOR}.matrix.gz \
                 $OPTIONS"
        # specify mean, varicance, and mixture weights of GMM used for Fisher vectors
        OPTIONS="--feature-quantization.fisher-vector.mean-file=gmm/mean.${DESCRIPTOR}.matrix.gz \
                 --feature-quantization.fisher-vector.variance-file=gmm/variance.${DESCRIPTOR}.matrix.gz \
                 --feature-quantization.fisher-vector.weights-file=gmm/weights.${DESCRIPTOR}.vector.gz \
                 --feature-quantization.ensure-sequence-length=$VIDEO_LEN \
                 $OPTIONS"
        # specify which feature cache to read and where to write the result
        OPTIONS="--features.feature-reader.feature-cache=results/${VIDEO_NAME}.${DESCRIPTOR}.cache \
                 --features.feature-writer.feature-cache=results/fv.${VIDEO_NAME}.${DESCRIPTOR}.cache \
                 $OPTIONS"
        run "../../../executables/feature-transformation --config=config/temporal-fisher-vectors.config $OPTIONS"
        # we do not need the trajectory file anymore
        rm results/${VIDEO_NAME}.${DESCRIPTOR}.cache
    done

    # concatenate the Fisher vectors computed on the five descriptor types
    OPTIONS="--log-file=log/combine.${VIDEO_NAME}.log \
             --action=cache-combination \
             --cache-combination.combination-method=concatenation \
             --cache-combination.number-of-caches=5 \
             --cache-combination.feature-reader-1.feature-cache=results/fv.${VIDEO_NAME}.traj.cache \
             --cache-combination.feature-reader-2.feature-cache=results/fv.${VIDEO_NAME}.hog.cache \
             --cache-combination.feature-reader-3.feature-cache=results/fv.${VIDEO_NAME}.hof.cache \
             --cache-combination.feature-reader-4.feature-cache=results/fv.${VIDEO_NAME}.mbhx.cache \
             --cache-combination.feature-reader-5.feature-cache=results/fv.${VIDEO_NAME}.mbhy.cache \
             --features.feature-writer.feature-cache=results/${VIDEO_NAME}.cache"
    run "../../../executables/feature-cache-manager $OPTIONS"
    # we do not need the descriptor based Fisher vectors anymore
    rm results/fv.${VIDEO_NAME}.*.cache

    ############################################################################

done
