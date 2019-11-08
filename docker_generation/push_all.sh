#!/usr/bin/env bash

docker push moinfar/bio-r-base:3.5.1
docker push moinfar/bio-python:3.6
docker push moinfar/bio-python:2.7
docker push moinfar/bio-deep-python:tensorflow-py36
docker push moinfar/bio-deep-python:pytorch-py36

docker push moinfar/sc-scimpute:latest
docker push moinfar/sc-magic:latest
docker push moinfar/sc-uncurl:latest
docker push moinfar/sc-biscuit:latest
docker push moinfar/sc-dca:latest
docker push moinfar/sc-dca:gpu
docker push moinfar/sc-decode:latest
docker push moinfar/sc-drimpute:latest
docker push moinfar/sc-knnsmoothing:latest
docker push moinfar/sc-saucie:latest
docker push moinfar/sc-saucie:gpu
docker push moinfar/sc-saver:latest
docker push moinfar/sc-scvi:latest
docker push moinfar/sc-scvi:gpu
docker push moinfar/sc-zinbwave:latest
docker push moinfar/sc-netsmooth:latest
docker push moinfar/sc-deepimpute:latest
docker push moinfar/sc-scscope:latest
docker push moinfar/sc-scscope:gpu
