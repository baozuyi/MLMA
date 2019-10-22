#!/usr/bin/env bash

if [ $# != 2 ] ; then 
    echo "USAGE: sh train.sh device config" 
    exit 1; 
fi 

CUDA_VISIBLE_DEVICES=$1 python train.py $2
