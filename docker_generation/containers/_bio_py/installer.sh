#!/usr/bin/env bash

apt-get update -y
apt-get install time -y

pip install --no-cache-dir pandas
pip install --no-cache-dir scipy
pip install --no-cache-dir scikit-learn
