#!/bin/bash

source ../../tools/tools.sh

mkdir -p log

# find all videos we want to use for detection
LIST=$(ls --color=none ../00_feature_extraction/results/video_test_*.cache)

for CACHE in $LIST; do
    LOG_FILE=$(echo $CACHE | sed -e "s#.*/##" | sed -e "s#\.cache##")

    OPTIONS="--log-file=log/$LOG_FILE.log \
             --features.feature-reader.feature-cache=$CACHE"

    run "../../../executables/action-detector --config=config/linearSearch.config $OPTIONS"
done

# convert result to human-readable output, i.e.
# write for each video the class and the start frame to detections.txt

LIST=$(ls --color=none log/*.log)
rm -f detections.txt

for LOG_FILE in $LIST; do
    echo "### $LOG_FILE ###" >> detections.txt
    # grep the line containing the detections from the config file
    DETECTIONS=$(tail -n3 $LOG_FILE | head -n1)
    for DET in $DETECTIONS; do
        # extract label and start frame (third field is log p(label|segment))
        LABEL=$(echo $DET | cut -f1 -d':')
        START_FRAME=$(echo $DET | cut -f2 -d':')
        # find the class name corresponding to the label
        if [ $LABEL -eq 0 ]; then # 0 is background
            CLS="background"
        else
            CLS=$(grep -e "^$LABEL " ../data/class-mapping.txt | cut -f2 -d' ')
        fi
        echo "$CLS $START_FRAME" >> detections.txt
    done
done
