#!/bin/bash

# Check if PA is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [Protected Attribute]"
    exit 1
fi

PA="$1"
idx=-1

if [ "$PA" == "sex" ]; then
    idx=8 # 8 stands for sex
else
    echo "Error: invalid PA provided for adult: sex"
    exit 1
fi

for ((i=1; i<=12; i++)); do # for each model 1 to 12
    echo -e "\n-----Running network AC-$i on $PA-----"
    ./network_test "../models/adult/AC-$i.nnet" "$idx"
done