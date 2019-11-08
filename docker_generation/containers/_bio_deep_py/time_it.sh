#!/usr/bin/env bash

/usr/bin/time -f "time in seconds: %e s\nkernel time: %S\nuser time: %U\nmax resident memory: %M KB\nexit status: %x" -o "$1" "${@:2}"
