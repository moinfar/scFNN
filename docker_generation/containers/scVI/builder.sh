#!/usr/bin/env bash

docker build -t moinfar/sc-scvi:latest .
docker build -f Dockerfile-gpu -t moinfar/sc-scvi:gpu .
