#!/bin/bash

################################################################################
### DECODING: RUN VITERBI ALORITHM ON EACH TEST SEQUENCE #######################
################################################################################

    START_ITER=0
    MAX_ITER=7 # also run this for 7 epochs, same as for training

    for ITER in $(seq $START_ITER $MAX_ITER); do

        sed -i "s#iter-[0-9]*#iter-$ITER#g" config/*.config
        N_HMM_STATES=$(head -n2 results/iter-${ITER}/train.labels | tail -n1 | cut -f2 -d' ')

        OPTIONS="--neural-network.output-layer.number-of-units=${N_HMM_STATES}"
        ../../src/Hmm/hmm-tool --config=config/decode.config $OPTIONS

    done

