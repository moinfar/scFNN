mkdir -p ~/code/test\ results/no-impute; cp ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/no-impute/random_mask_cell_cycle_1.csv

~/code/single-cell-IMP-methods/run_scripts/dca-gpu.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/dca/random_mask_cell_cycle_1.csv ~/code/test\ results/dca/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/deepimpute.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/deepimpute/random_mask_cell_cycle_1.csv ~/code/test\ results/deepimpute/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/drimpute.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/drimpute/random_mask_cell_cycle_1.csv ~/code/test\ results/drimpute/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/knn-smoothing.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/knn-smoothing/random_mask_cell_cycle_1.csv ~/code/test\ results/knn-smoothing/random_mask_cell_cycle_1.csv.d -k 15
~/code/single-cell-IMP-methods/run_scripts/magic.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/magic/random_mask_cell_cycle_1.csv ~/code/test\ results/magic/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/saucie-gpu.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/saucie/random_mask_cell_cycle_1.csv ~/code/test\ results/saucie/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/saver.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/saver/random_mask_cell_cycle_1.csv ~/code/test\ results/saver/random_mask_cell_cycle_1.csv.d --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scimpute.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/scimpute/random_mask_cell_cycle_1.csv ~/code/test\ results/scimpute/random_mask_cell_cycle_1.csv.d --Kcluster 50 --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scscope-gpu.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/scscope/random_mask_cell_cycle_1.csv ~/code/test\ results/scscope/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/scvi-gpu.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/scvi/random_mask_cell_cycle_1.csv ~/code/test\ results/scvi/random_mask_cell_cycle_1.csv.d
~/code/single-cell-IMP-methods/run_scripts/uncurl.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/uncurl/random_mask_cell_cycle_1.csv ~/code/test\ results/uncurl/random_mask_cell_cycle_1.csv.d --clusters 50
~/code/single-cell-IMP-methods/run_scripts/zinbwave.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/zinbwave/random_mask_cell_cycle_1.csv ~/code/test\ results/zinbwave/random_mask_cell_cycle_1.csv.d --ncores 8 --gene_subset 1000

# ~/code/single-cell-IMP-methods/run_scripts/biscuit.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/biscuit/random_mask_cell_cycle_1.csv ~/code/test\ results/biscuit/random_mask_cell_cycle_1.csv.d --ncores 8
# ~/code/single-cell-IMP-methods/run_scripts/netsmooth.sh ~/code/test\ benchs/random_mask_cell_cycle_1.csv ~/code/test\ results/netsmooth/random_mask_cell_cycle_1.csv ~/code/test\ results/netsmooth/random_mask_cell_cycle_1.csv.d --ncores 8

