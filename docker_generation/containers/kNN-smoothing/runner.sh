#!/usr/bin/env bash

# i/o
export inputpath=$1
export outdir=$2

mkdir "$outdir"
bash /time_it.sh /tmp/time.txt python3 knn_smooth.py -f "$inputpath" -o "$outdir/knn_smoothed.csv" --sep , ${@:3}
cp /tmp/time.txt "$outdir/time.txt"
