#!/usr/bin/env bash

docker build -f Dockerfile-tensorflow -t moinfar/bio-deep-python:tensorflow-py36 .
docker build -f Dockerfile-pytorch -t moinfar/bio-deep-python:pytorch-py36 .
