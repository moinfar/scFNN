#!/usr/bin/env bash

apt-get update -y
apt-get install time -y

R -e 'install.packages("optparse")'
