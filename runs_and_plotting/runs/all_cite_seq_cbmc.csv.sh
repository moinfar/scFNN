mkdir -p ~/code/test\ results/no-impute; cp ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/no-impute/cite_seq_cbmc.csv

~/code/single-cell-IMP-methods/run_scripts/dca-gpu.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/dca/cite_seq_cbmc.csv ~/code/test\ results/dca/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/deepimpute.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/deepimpute/cite_seq_cbmc.csv ~/code/test\ results/deepimpute/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/drimpute.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/drimpute/cite_seq_cbmc.csv ~/code/test\ results/drimpute/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/knn-smoothing.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/knn-smoothing/cite_seq_cbmc.csv ~/code/test\ results/knn-smoothing/cite_seq_cbmc.csv.d -k 15
~/code/single-cell-IMP-methods/run_scripts/magic.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/magic/cite_seq_cbmc.csv ~/code/test\ results/magic/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/saucie-gpu.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/saucie/cite_seq_cbmc.csv ~/code/test\ results/saucie/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/saver.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/saver/cite_seq_cbmc.csv ~/code/test\ results/saver/cite_seq_cbmc.csv.d --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scimpute.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/scimpute/cite_seq_cbmc.csv ~/code/test\ results/scimpute/cite_seq_cbmc.csv.d --Kcluster 50 --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scscope-gpu.sh  ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/scscope/cite_seq_cbmc.csv ~/code/test\ results/scscope/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/scvi-gpu.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/scvi/cite_seq_cbmc.csv ~/code/test\ results/scvi/cite_seq_cbmc.csv.d
~/code/single-cell-IMP-methods/run_scripts/uncurl.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/uncurl/cite_seq_cbmc.csv ~/code/test\ results/uncurl/cite_seq_cbmc.csv.d --clusters 50
~/code/single-cell-IMP-methods/run_scripts/zinbwave.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/zinbwave/cite_seq_cbmc.csv ~/code/test\ results/zinbwave/cite_seq_cbmc.csv.d --ncores 8 --gene_subset 1000

# ~/code/single-cell-IMP-methods/run_scripts/biscuit.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/biscuit/cite_seq_cbmc.csv ~/code/test\ results/biscuit/cite_seq_cbmc.csv.d --ncores 8
# ~/code/single-cell-IMP-methods/run_scripts/netsmooth.sh ~/code/test\ benchs/cite_seq_cbmc.csv ~/code/test\ results/netsmooth/cite_seq_cbmc.csv ~/code/test\ results/netsmooth/cite_seq_cbmc.csv.d --ncores 8

