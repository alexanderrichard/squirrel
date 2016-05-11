#!/bin/bash

# create label file and bundle file for training data
# files can be found in features/

# create train.bundle/test.bundle and labels.train.cache/labels.test.cache
for PART in train test; do

    echo "#bundle" > features/${PART}.bundle

    while read LINE; do
        VIDEO=$(echo $LINE | cut -f1 -d' ')
        LABEL=$(echo $LINE | cut -f2 -d' ')
        echo "features/caches/${VIDEO}.traj.cache" >> features/${PART}.bundle
        echo $LABEL >> features/labels.tmp
    done < features/${PART}.videos+labels

    # create label cache from file features/labels.tmp
    OPTIONS="--log-file=/dev/null \
             --action=external-to-cache \
             --external-type=ascii-labels \
             --converter.label-converter.label-file=features/labels.tmp \
             --features.label-writer.feature-cache=features/labels.${PART}.cache"
    ../../../executables/converter $OPTIONS

done

rm -f features/labels.tmp
