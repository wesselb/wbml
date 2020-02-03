#!/usr/bin/env zsh
set -e
rm Bra* Cam* Sot* Chi*
for day in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31; do
    wget http://www.sotonmet.co.uk/archive/2013/July/CSV/Sot${day}Jul2013.csv
    wget http://www.bramblemet.co.uk/archive/2013/July/CSV/Bra${day}Jul2013.csv
    wget http://www.chimet.co.uk/archive/2013/July/CSV/Chi${day}Jul2013.csv
    wget http://www.cambermet.co.uk/archive/2013/July/CSV/Cam${day}Jul2013.csv
done
