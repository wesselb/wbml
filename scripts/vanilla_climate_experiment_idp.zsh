#!/usr/bin/env zsh

for ((i = 0; i <= 28; i++)); do
    python scripts/climate_independent_gps.py --model $i $@
done
