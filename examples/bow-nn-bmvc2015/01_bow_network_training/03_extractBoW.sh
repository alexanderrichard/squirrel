#!/bin/bash

mkdir -p results
mkdir -p log

# extract nn-bow features
for PART in train test; do
    OPTIONS="--log-file=log/nn-features.${PART}.${DESCR}.log \
             --features.feature-reader.feature-cache=features/$PART.bundle \
             --features.feature-writer.feature-cache=results/${PART}.bow.cache"
    
    ../../../executables/feature-transformation --config=config/extractBoW.config $OPTIONS
done

# transform to ascii features (for further processing, e.g. with libsvm)
for PART in train test; do
    CACHE=results/${PART}.bow.cache
    ASCII_FILE=results/${PART}.bow.txt
    OPTIONS="--log-file=/dev/null \
             --action=print-cache \
             --features.feature-reader.feature-cache=$CACHE"
    ../../../executables/feature-cache-manager $OPTIONS | tail -n+3 > $ASCII_FILE
done
