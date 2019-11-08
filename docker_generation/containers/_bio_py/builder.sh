#!/usr/bin/env bash

echo "building for python version ${pyver-3.6}."
docker build --build-arg pyver=${pyver-3.6} -t moinfar/bio-python:${pyver-3.6} .
