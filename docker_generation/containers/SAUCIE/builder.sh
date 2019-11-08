#!/usr/bin/env bash

docker build -t moinfar/sc-saucie:latest .
docker build -f Dockerfile-gpu -t moinfar/sc-saucie:gpu .
