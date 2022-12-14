#!/bin/bash

DATASET=opera_RBioDeg

for FEATURE in mordred mfp
do
	python make_bo_traces.py --dataset=$i
done

