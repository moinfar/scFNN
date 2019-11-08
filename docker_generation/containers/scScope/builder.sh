#!/usr/bin/env bash

docker build -t moinfar/sc-scscope:latest .
docker build -f Dockerfile-gpu -t moinfar/sc-scscope:gpu .
