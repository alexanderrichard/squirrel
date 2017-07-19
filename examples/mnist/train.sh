#!/bin/bash

mkdir -p log results

../../src/Nn/neural-network-trainer --config=config/training.config
