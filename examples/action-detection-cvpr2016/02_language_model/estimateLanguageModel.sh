#!/bin/bash

source ../../tools/tools.sh

mkdir -p log
mkdir -p results

NUMBER_OF_CLASSES=21 # 20 action classes and a background class
NGRAM="trigram"

# create a file containing "sentences", ie the sequence of actions for each video
rm -f results/lm.validation.corpus
VIDEO_LIST=$(ls --color=none ../data/framewise-labels/*validation*) # use validation set of Thumos for language model estimation
for VIDEO in $VIDEO_LIST; do
    echo $(uniq $VIDEO) >> results/lm.validation.corpus
done

# estimate language model
OPTIONS="--log-file=log/$NGRAM.log \
         --action=build \
         --corpus=results/lm.validation.corpus \
         --language-model.lexicon-size=$NUMBER_OF_CLASSES \
         --language-model.file=results/$NGRAM.lm \
         --language-model.type=$NGRAM"

run "../../../executables/language-model $OPTIONS"

echo -e "\nLanguage model estimated.\n"

# compute perplexity
OPTIONS="--action=perplexity \
         --log-file=log/perplexity.$NGRAM.log \
         --corpus=results/lm.validation.corpus \
         --language-model.lexicon-size=$NUMBER_OF_CLASSES \
         --language-model.file=results/$NGRAM.lm \
         --language-model.type=$NGRAM \
         --language-model.backing-off=true"

run "../../../executables/language-model $OPTIONS"

echo -e "\nPerplexity computed. See log/perplexity.$NGRAM.log for results.\n"
