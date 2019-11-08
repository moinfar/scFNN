#!/usr/bin/env bash

docker pull moinfar/bio-r-base:3.5.1
docker pull moinfar/bio-python:3.6
docker pull moinfar/bio-python:2.7
docker pull moinfar/bio-deep-python:tensorflow-py36
docker pull moinfar/bio-deep-python:pytorch-py36

docker pull moinfar/sc-scimpute:latest
docker pull moinfar/sc-magic:latest
docker pull moinfar/sc-uncurl:latest
docker pull moinfar/sc-biscuit:latest
docker pull moinfar/sc-dca:latest
docker pull moinfar/sc-dca:gpu
docker pull moinfar/sc-decode:latest
docker pull moinfar/sc-drimpute:latest
docker pull moinfar/sc-knnsmoothing:latest
docker pull moinfar/sc-saucie:latest
docker pull moinfar/sc-saucie:gpu
docker pull moinfar/sc-saver:latest
docker pull moinfar/sc-scvi:latest
docker pull moinfar/sc-scvi:gpu
docker pull moinfar/sc-zinbwave:latest
docker pull moinfar/sc-netsmooth:latest
docker pull moinfar/sc-deepimpute:latest
docker pull moinfar/sc-scscope:latest
docker pull moinfar/sc-scscope:gpu
