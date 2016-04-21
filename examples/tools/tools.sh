#/bin/bash

#$1: command to evaluate
function run {
    local CMD=$(echo "$1" | tr -s ' ')
    echo "$CMD"
    eval "$CMD"
}
