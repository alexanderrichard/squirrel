#!/bin/bash

source ../tools/tools.sh

mkdir -p log
mkdir -p results

run "../../executables/neural-network-trainer --config=config/training.config"
