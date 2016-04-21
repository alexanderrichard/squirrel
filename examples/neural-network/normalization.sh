#!/bin/bash

source ../tools/tools.sh

mkdir -p log
mkdir -p normalization

run "../../executables/feature-transformation --config=config/normalization.config"
