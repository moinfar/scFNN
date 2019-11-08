mkdir -p ~/code/test\ results/no-impute; cp ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/no-impute/paired_data_pollen.csv

~/code/single-cell-IMP-methods/run_scripts/dca-gpu.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/dca/paired_data_pollen.csv ~/code/test\ results/dca/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/deepimpute.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/deepimpute/paired_data_pollen.csv ~/code/test\ results/deepimpute/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/drimpute.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/drimpute/paired_data_pollen.csv ~/code/test\ results/drimpute/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/knn-smoothing.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/knn-smoothing/paired_data_pollen.csv ~/code/test\ results/knn-smoothing/paired_data_pollen.csv.d -k 15
~/code/single-cell-IMP-methods/run_scripts/magic.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/magic/paired_data_pollen.csv ~/code/test\ results/magic/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/saucie-gpu.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/saucie/paired_data_pollen.csv ~/code/test\ results/saucie/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/saver.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/saver/paired_data_pollen.csv ~/code/test\ results/saver/paired_data_pollen.csv.d --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scimpute.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/scimpute/paired_data_pollen.csv ~/code/test\ results/scimpute/paired_data_pollen.csv.d --Kcluster 13 --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scscope-gpu.sh  ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/scscope/paired_data_pollen.csv ~/code/test\ results/scscope/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/scvi-gpu.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/scvi/paired_data_pollen.csv ~/code/test\ results/scvi/paired_data_pollen.csv.d
~/code/single-cell-IMP-methods/run_scripts/uncurl.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/uncurl/paired_data_pollen.csv ~/code/test\ results/uncurl/paired_data_pollen.csv.d --clusters 13
~/code/single-cell-IMP-methods/run_scripts/zinbwave.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/zinbwave/paired_data_pollen.csv ~/code/test\ results/zinbwave/paired_data_pollen.csv.d --ncores 8 --gene_subset 1000

# ~/code/single-cell-IMP-methods/run_scripts/biscuit.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/biscuit/paired_data_pollen.csv ~/code/test\ results/biscuit/paired_data_pollen.csv.d --ncores 8
# ~/code/single-cell-IMP-methods/run_scripts/netsmooth.sh ~/code/test\ benchs/paired_data_pollen.csv ~/code/test\ results/netsmooth/paired_data_pollen.csv ~/code/test\ results/netsmooth/paired_data_pollen.csv.d --ncores 8

