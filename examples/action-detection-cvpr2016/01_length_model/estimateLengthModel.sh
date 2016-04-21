#!/bin/bash

source ../../tools/tools.sh

mkdir -p log
mkdir -p results

NUMBER_OF_CLASSES=21 # 20 action classes and a background class

# create a file containing lengths for all segments
rm -f results/length-file
VIDEO_LIST=$(ls --color=none ../data/framewise-labels/*validation*) # use validation set of Thumos for length model estimation
for VIDEO in $VIDEO_LIST; do
    uniq -c $VIDEO >> results/length-file
done

OPTIONS="--log-file=log/length-model.log \
         --action=estimate-poisson-model \
         --length-model.training-file=results/length-file \
         --length-model.number-of-classes=${NUMBER_OF_CLASSES} \
         --length-model.file=results/poisson-model.gz"

run "../../../executables/action-detector $OPTIONS"
