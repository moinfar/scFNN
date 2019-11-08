#!/usr/bin/env bash

# r-base:3.4 with bioconductor
cd containers/_bio_r_base
export rver=3.5.1
sh ./builder.sh
cd ../..

# python:3.6 with some good stuff (like pandas)
cd containers/_bio_py
export pyver=3.6
sh ./builder.sh
cd ../..

# python:2.7 with some good stuff (like pandas)
cd containers/_bio_py
export pyver=2.7
sh ./builder.sh
cd ../..

# python:3.6 with deep learning libraries
cd containers/_bio_deep_py
sh ./builder.sh
cd ../..

# scimpute
cd containers/scImpute
sh ./builder.sh
cd ../..

# MAGIC
cd containers/MAGIC
sh ./builder.sh
cd ../..

# UNCURL
cd containers/UNCURL
sh ./builder.sh
cd ../..

# BISCUIT
cd containers/BISCUIT
sh ./builder.sh
cd ../..

# DCA
cd containers/DCA
sh ./builder.sh
cd ../..

# DECODE
cd containers/DECODE
sh ./builder.sh
cd ../..

# DrImpute
cd containers/DrImpute
sh ./builder.sh
cd ../..

# kNN-smoothing
cd containers/kNN-smoothing
sh ./builder.sh
cd ../..

# SAUCIE
cd containers/SAUCIE
sh ./builder.sh
cd ../..

# SAVER
cd containers/SAVER
sh ./builder.sh
cd ../..

# scVI
cd containers/scVI
sh ./builder.sh
cd ../..

# ZINB-WaVe
cd containers/ZINB-WaVe
sh ./builder.sh
cd ../..

# netSmooth
cd containers/netSmooth
sh ./builder.sh
cd ../..

# deepImpute
cd containers/deepImpute
sh ./builder.sh
cd ../..

# scScope
cd containers/scScope
sh ./builder.sh
cd ../..
