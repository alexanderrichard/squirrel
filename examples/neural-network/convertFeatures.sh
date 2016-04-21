#!/bin/bash

# convert features to squirrel caches

source ../tools/tools.sh

for PART in test train; do
    # convert data matrix
    zcat features/${PART}.matrix.gz > features/${PART}.matrix
    OPTIONS="--log-file=/dev/null \
             --action=internal-cache-conversion \
             --internal-cache-conversion-type=matrix-to-single-cache \
             --converter.matrix-to-cache-converter.matrix-file=features/${PART}.matrix \
             --features.feature-writer.feature-cache=features/${PART}.cache"
    run "../../executables/converter $OPTIONS"

    # convert label file
    OPTIONS="--log-file=/dev/null \
             --action=external-to-cache \
             --external-type=ascii-labels \
             --converter.label-converter.label-file=features/${PART}.labels \
             --features.label-writer.feature-cache=features/labels.${PART}.cache"
    run "../../executables/converter $OPTIONS"

    # remove ascii files
    rm features/${PART}.matrix
done
