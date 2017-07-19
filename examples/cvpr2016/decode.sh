#!/bin/bash

mkdir -p log results

### TRIGRAM LANGUAGE MODEL (based on validation set) ###
../../src/Hmm/hmm-tool --config=config/grammar.config

### DECODING OF A TEST SEQUENCE                      ###
../../src/Hmm/hmm-tool --config=config/decode.config

