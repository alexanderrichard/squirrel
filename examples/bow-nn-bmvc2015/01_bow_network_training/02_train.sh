#!/bin/bash

mkdir -p results
mkdir -p log

# train the network (as specified in config/training.config)
../../../executables/neural-network-trainer --config=config/training.config
