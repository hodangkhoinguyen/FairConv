#!/bin/bash

# Check if PA is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [Protected Attribute]"
    exit 1
fi

PA="$1"
idx=-1

if [ "$PA" == "age" ]; then
    idx=0 # 0 stands for age
else
    echo "Error: invalid PA provided for bank: age"
    exit 1
fi

for ((i=1; i<=8; i++)); do # for each model 1 to 8
    echo -e "\n-----Running network BM-$i on $PA-----"
    ./network_test "../models/bank/BM-$i.nnet" "$idx"
done