mkdir -p ~/code/test\ results/no-impute; cp ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/no-impute/down_sample_cite_cd8_1.csv

~/code/single-cell-IMP-methods/run_scripts/dca-gpu.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/dca/down_sample_cite_cd8_1.csv ~/code/test\ results/dca/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/deepimpute.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/deepimpute/down_sample_cite_cd8_1.csv ~/code/test\ results/deepimpute/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/drimpute.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/drimpute/down_sample_cite_cd8_1.csv ~/code/test\ results/drimpute/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/knn-smoothing.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/knn-smoothing/down_sample_cite_cd8_1.csv ~/code/test\ results/knn-smoothing/down_sample_cite_cd8_1.csv.d -k 15
~/code/single-cell-IMP-methods/run_scripts/magic.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/magic/down_sample_cite_cd8_1.csv ~/code/test\ results/magic/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/saucie-gpu.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/saucie/down_sample_cite_cd8_1.csv ~/code/test\ results/saucie/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/saver.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/saver/down_sample_cite_cd8_1.csv ~/code/test\ results/saver/down_sample_cite_cd8_1.csv.d --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scimpute.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/scimpute/down_sample_cite_cd8_1.csv ~/code/test\ results/scimpute/down_sample_cite_cd8_1.csv.d --Kcluster 50 --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scscope-gpu.sh  ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/scscope/down_sample_cite_cd8_1.csv ~/code/test\ results/scscope/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/scvi-gpu.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/scvi/down_sample_cite_cd8_1.csv ~/code/test\ results/scvi/down_sample_cite_cd8_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/uncurl.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/uncurl/down_sample_cite_cd8_1.csv ~/code/test\ results/uncurl/down_sample_cite_cd8_1.csv.d --clusters 50
~/code/single-cell-IMP-methods/run_scripts/zinbwave.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/zinbwave/down_sample_cite_cd8_1.csv ~/code/test\ results/zinbwave/down_sample_cite_cd8_1.csv.d --ncores 8 --gene_subset 1000

# ~/code/single-cell-IMP-methods/run_scripts/biscuit.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/biscuit/down_sample_cite_cd8_1.csv ~/code/test\ results/biscuit/down_sample_cite_cd8_1.csv.d --ncores 8
# ~/code/single-cell-IMP-methods/run_scripts/netsmooth.sh ~/code/test\ benchs/down_sample_cite_cd8_1.csv ~/code/test\ results/netsmooth/down_sample_cite_cd8_1.csv ~/code/test\ results/netsmooth/down_sample_cite_cd8_1.csv.d --ncores 8

