#!/usr/bin/env zsh

for ((i = 10; i <= 28; i++)); do
    python scripts/climate.py --model $i $@
done
