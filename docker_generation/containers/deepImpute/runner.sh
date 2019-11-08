#!/usr/bin/env bash

# i/o
export inputpath=$1
export outdir=$2

mkdir $outdir

bash /time_it.sh /tmp/time.txt python /app/deepimpute/deepImpute.py --cell-axis columns -o "$outdir/deepimpute.csv" "$inputpath" "${@:3}"
cp /tmp/time.txt "$outdir/time.txt"

python transpose_it.py -i "$outdir/deepimpute.csv" -o "$outdir/deepimpute.csv"
