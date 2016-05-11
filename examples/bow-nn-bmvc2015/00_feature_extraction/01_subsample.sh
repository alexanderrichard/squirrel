#!/bin/bash

mkdir -p subsampled-caches
mkdir -p log

# max number of feature vectors per sequence
MAX_SAMPLES=5000

# list of all squirrel caches in caches/
CACHE_LIST=$(ls --color=none caches/*.cache)

# subsample all squirrel caches in caches/
for CACHE in $CACHE_LIST; do
    NAME=$(echo $CACHE | sed -e "s#.*/##" | sed -e "s#\.cache##")
    OPTIONS="--log-file=log/subsampling.$NAME.log \
             --action=subsampling \
             --features.feature-cache-manager.max-number-of-samples=$MAX_SAMPLES \
             --features.feature-cache-manager.sampling-mode=uniform \
             --features.feature-reader.feature-cache=caches/$NAME.cache \
             --features.feature-writer.feature-cache=subsampled-caches/$NAME.cache"
    ../../../executables/feature-cache-manager $OPTIONS
done

