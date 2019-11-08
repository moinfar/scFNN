#!/usr/bin/env bash

# i/o
export outputpath=$2

bash /time_it.sh /tmp/time.txt dca "$@"
cp /tmp/time.txt "$outputpath/time.txt"

python tsv_to_csv.py < "$outputpath/mean.tsv" > "$outputpath/mean.csv"
