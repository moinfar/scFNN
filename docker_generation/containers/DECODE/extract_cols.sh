#!/usr/bin/env bash

# i/o
export inputpath=$1
export cols=$2

head -n 1 $inputpath | sed 's/,/\n/g' | sed -r '/^\s*$/d' > $cols
