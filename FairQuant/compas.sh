#!/bin/bash

# Check if PA is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [Protected Attribute]"
    exit 1
fi

PA="$1"
idx=-1

if [ "$PA" == "age" ]; then
    idx=2 # 2 stands for age
elif [ "$PA" == "race" ]; then
    idx=3 # 3 stands for race ---- SAME AS PAPER
elif [ "$PA" == "sex" ]; then
    idx=4 # 4 stands for sex
else
    echo "Error: invalid PA provided for compas: age, race, or sex"
    exit 1
fi

for ((i=1; i<=7; i++)); do # for each model 1 to 7
    echo -e "\n-----Running network compas-$i on $PA-----"
    ./network_test "../models/compas/compas-$i.nnet" "$idx"
done