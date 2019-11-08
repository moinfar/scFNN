#!/usr/bin/env bash

echo "building for R version ${rver-3.5.1}."
docker build --build-arg rver=${rver-3.5.1} -t moinfar/bio-r-base:${rver-3.5.1} .
