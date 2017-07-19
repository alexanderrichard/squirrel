#!/bin/bash

################################################################################
### INITIALIZATION: GENERATE GRAMMAR AND LINEAR ALIGNMENT ######################
################################################################################

    SPLIT=split1 # this setup is for split1 of the Breakfast dataset
    RESULT_DIR=results/iter-0
    mkdir -p $RESULT_DIR log

    ### GRAMMAR ################################################################
    ../../src/Hmm/hmm-tool --config=config/grammar.config

    ### INITIAL HMM ############################################################
    FRAMES_PER_STATE=10
    N_CLASSES=$(head -n2 data/${SPLIT}.train.transcripts | tail -n1 | cut -f2 -d' ')
    N_FRAMES=$(head -n2 data/${SPLIT}.train.labels | tail -n1 | cut -f1 -d' ')
    N_INSTANCES=$(head -n2 data/${SPLIT}.train.transcripts | tail -n1 | cut -f1 -d' ')
    STATES_PER_CLASS=`python -c "import math; print int( math.ceil( float(${N_FRAMES}) / ${N_INSTANCES} / ${FRAMES_PER_STATE} ) )"`
    echo "$N_CLASSES" > ${RESULT_DIR}/hmm_definition.vector
    for X in $(seq 1 $N_CLASSES); do echo "$STATES_PER_CLASS" >> ${RESULT_DIR}/hmm_definition.vector ; done

    ### LINEAR ALIGNMENT #######################################################
    ./utils/linear_alignment.py data/${SPLIT}.train.labels data/${SPLIT}.train.transcripts ${RESULT_DIR}/hmm_definition.vector ${RESULT_DIR}/train.labels


################################################################################
### TRAINING: ITERATE BETWEEN RNN TRAINING AND REESTIMATION ####################
################################################################################

    START_ITER=0
    MAX_ITER=7 # iterate for 7 epochs

    for ITER in $(seq $START_ITER $MAX_ITER); do

        sed -i "s#iter-[0-9]*#iter-$ITER#g" config/*.config

        RESULT_DIR=results/iter-$ITER
        mkdir -p log ${RESULT_DIR}

        N_HMM_STATES=$(head -n2 ${RESULT_DIR}/train.labels | tail -n1 | cut -f2 -d' ')

        ### RNN TRAINING #######################################################
        OPTIONS="--neural-network.output-layer.number-of-units=$N_HMM_STATES"
        ../../src/Nn/neural-network-trainer --config=config/training.config $OPTIONS

        # get best epoch (based on classification error on training set)
        BEST_EPOCH=$(grep "average classification error in epoch" log/training.iter-$ITER.log | sed -e "s#.*epoch ##" | sort -n -k2 | head -n1 | sed -e "s#:.*##")
        # only keep weights of best epoch
        for MODEL_FILE in $(ls --color=none ${RESULT_DIR}/*epoch-$BEST_EPOCH.*); do
            FINAL_MODEL_FILE=$(echo $MODEL_FILE | sed -e "s#\.epoch-$BEST_EPOCH##")
            mv $MODEL_FILE $FINAL_MODEL_FILE
        done
        rm ${RESULT_DIR}/*epoch*

        ### TRANSITION PROBABILITY AND STATE PRIOR #############################
        ./utils/transition_probabilities.py ${RESULT_DIR}/train.labels ${RESULT_DIR}/transition_probabilities.vector
        ./utils/prior.py ${RESULT_DIR}/train.labels ${RESULT_DIR}/prior.vector

        ### REALIGNMENT ########################################################
        OPTIONS="--neural-network.output-layer.number-of-units=${N_HMM_STATES}"
        ../../src/Hmm/hmm-tool --config=config/realign.config $OPTIONS

        # create new hmm file and alignment for next iteration
        FRAMES_PER_STATE=10
        if [ $ITER -lt $MAX_ITER ]; then
            NEXT_ITER=$(( $ITER + 1 ))
            mkdir -p results/iter-${NEXT_ITER}
            N_CLASSES=$(head -n1 ${RESULT_DIR}/hmm_definition.vector)
            ./utils/reestimate_hmm.py log/realign.iter-${ITER}.log $N_CLASSES $FRAMES_PER_STATE results/iter-${NEXT_ITER}/hmm_definition.vector results/iter-${NEXT_ITER}/train.labels
        fi

    done
