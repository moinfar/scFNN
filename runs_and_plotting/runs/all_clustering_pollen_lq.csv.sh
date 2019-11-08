mkdir -p ~/code/test\ results/no-impute; cp ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/no-impute/clustering_pollen_lq.csv

~/code/single-cell-IMP-methods/run_scripts/dca-gpu.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/dca/clustering_pollen_lq.csv ~/code/test\ results/dca/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/deepimpute.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/deepimpute/clustering_pollen_lq.csv ~/code/test\ results/deepimpute/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/drimpute.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/drimpute/clustering_pollen_lq.csv ~/code/test\ results/drimpute/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/knn-smoothing.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/knn-smoothing/clustering_pollen_lq.csv ~/code/test\ results/knn-smoothing/clustering_pollen_lq.csv.d -k 15
~/code/single-cell-IMP-methods/run_scripts/magic.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/magic/clustering_pollen_lq.csv ~/code/test\ results/magic/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/saucie-gpu.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/saucie/clustering_pollen_lq.csv ~/code/test\ results/saucie/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/saver.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/saver/clustering_pollen_lq.csv ~/code/test\ results/saver/clustering_pollen_lq.csv.d --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scimpute.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/scimpute/clustering_pollen_lq.csv ~/code/test\ results/scimpute/clustering_pollen_lq.csv.d --Kcluster 50 --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scscope-gpu.sh  ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/scscope/clustering_pollen_lq.csv ~/code/test\ results/scscope/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/scvi-gpu.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/scvi/clustering_pollen_lq.csv ~/code/test\ results/scvi/clustering_pollen_lq.csv.d
~/code/single-cell-IMP-methods/run_scripts/uncurl.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/uncurl/clustering_pollen_lq.csv ~/code/test\ results/uncurl/clustering_pollen_lq.csv.d --clusters 50
~/code/single-cell-IMP-methods/run_scripts/zinbwave.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/zinbwave/clustering_pollen_lq.csv ~/code/test\ results/zinbwave/clustering_pollen_lq.csv.d --ncores 8 --gene_subset 1000

# ~/code/single-cell-IMP-methods/run_scripts/biscuit.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/biscuit/clustering_pollen_lq.csv ~/code/test\ results/biscuit/clustering_pollen_lq.csv.d --ncores 8
# ~/code/single-cell-IMP-methods/run_scripts/netsmooth.sh ~/code/test\ benchs/clustering_pollen_lq.csv ~/code/test\ results/netsmooth/clustering_pollen_lq.csv ~/code/test\ results/netsmooth/clustering_pollen_lq.csv.d --ncores 8


