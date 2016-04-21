#!/bin/bash

source ../tools/tools.sh

mkdir -p log

run "../../executables/neural-network-trainer --config=config/recognition.config"
