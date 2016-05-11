#!/bin/bash

mkdir -p normalization
mkdir -p log

# compute mean and standard deviation for z-score normalization of data
OPTIONS="--log-file=log/normalization.estimate.log \
         --action=mean-and-variance-estimation \
         --feature-transformation.mean-and-variance-estimation.mean-file=normalization/mean.vector.gz \
         --feature-transformation.mean-and-variance-estimation.standard-deviation-file=normalization/standard-deviation.vector.gz \
         --features.feature-reader.feature-cache=features/train.bundle"

../../../executables/feature-transformation $OPTIONS

