#!/bin/bash

if [ $# -lt 3 ]; then
    echo "Usage: $0 <game> <input> <output>"
    exit 1
fi
game=$1
input=$2
output=$3

if [ "$game" == "cave" ]; then
    reachmove="maze"
fi
if [ "$game" == "platform" ]; then
    reachmove="platform"
fi
if [ "$game" == "slide" ]; then
    reachmove="tomb"
fi
if [ "$game" == "vertical" ]; then
    reachmove="supercat-new"
fi
if [ "$game" == "cavedoors" ]; then
    reachmove="maze"
fi
if [ "$game" == "caveportal" ]; then
    reachmove="maze"
fi

python ./sturgeon/level2repath.py --textfile $input --outfile $output --reach-connect "--src { --dst } --move $reachmove"