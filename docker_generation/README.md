# single-cell-IMP-methods


### Running

Command to run each algorithm:

```
# scImpute
./run_scripts/scimpute.sh input_file.csv output_file.csv output_dir
# MAGIC
./run_scripts/magic.sh input_file.csv output_file.csv output_dir
# UNCURL
./run_scripts/uncurl.sh input_file.csv output_file.csv output_dir
# BISCUIT (is very buggy)
./run_scripts/uncurl.sh input_file.csv output_file.csv output_dir
# DCA (cpu version only)
./run_scripts/dca-cpu.sh input_file.csv output_file.csv output_dir
# DECODE (needs verification)
./run_scripts/decode.sh input_file.csv output_file.csv output_dir
# DrImpute
./run_scripts/drimpute.sh input_file.csv output_file.csv output_dir
# kNN-smoothing
./run_scripts/knn-smoothing.sh input_file.csv output_file.csv output_dir
# deepImpute
./run_scripts/knn-deepimpute.sh input_file.csv output_file.csv output_dir
# netsmooth
./run_scripts/knn-deepimpute.sh input_file.csv output_file.csv output_dir
```


### Re-Building

There is no need to build images locally.
However, To build docker images run:

```
./build_all.sh
```

### Pulling

To pull all docker images once, execute:

```
./pull_all.sh
```
