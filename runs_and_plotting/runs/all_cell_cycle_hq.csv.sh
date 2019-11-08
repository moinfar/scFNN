mkdir -p ~/code/test\ results/no-impute; cp ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/no-impute/cell_cycle_hq.csv

~/code/single-cell-IMP-methods/run_scripts/dca-gpu.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/dca/cell_cycle_hq.csv ~/code/test\ results/dca/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/deepimpute.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/deepimpute/cell_cycle_hq.csv ~/code/test\ results/deepimpute/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/drimpute.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/drimpute/cell_cycle_hq.csv ~/code/test\ results/drimpute/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/knn-smoothing.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/knn-smoothing/cell_cycle_hq.csv ~/code/test\ results/knn-smoothing/cell_cycle_hq.csv.d -k 15
~/code/single-cell-IMP-methods/run_scripts/magic.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/magic/cell_cycle_hq.csv ~/code/test\ results/magic/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/saucie-gpu.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/saucie/cell_cycle_hq.csv ~/code/test\ results/saucie/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/saver.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/saver/cell_cycle_hq.csv ~/code/test\ results/saver/cell_cycle_hq.csv.d --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scimpute.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/scimpute/cell_cycle_hq.csv ~/code/test\ results/scimpute/cell_cycle_hq.csv.d --Kcluster 3 --ncores 8
~/code/single-cell-IMP-methods/run_scripts/scscope-gpu.sh  ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/scscope/cell_cycle_hq.csv ~/code/test\ results/scscope/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/scvi-gpu.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/scvi/cell_cycle_hq.csv ~/code/test\ results/scvi/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/uncurl.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/uncurl/cell_cycle_hq.csv ~/code/test\ results/uncurl/cell_cycle_hq.csv.d
~/code/single-cell-IMP-methods/run_scripts/zinbwave.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/zinbwave/cell_cycle_hq.csv ~/code/test\ results/zinbwave/cell_cycle_hq.csv.d --ncores 8

# ~/code/single-cell-IMP-methods/run_scripts/biscuit.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/biscuit/cell_cycle_hq.csv ~/code/test\ results/biscuit/cell_cycle_hq.csv.d
# ~/code/single-cell-IMP-methods/run_scripts/netsmooth.sh ~/code/test\ benchs/cell_cycle_hq.csv ~/code/test\ results/netsmooth/cell_cycle_hq.csv ~/code/test\ results/netsmooth/cell_cycle_hq.csv.d
