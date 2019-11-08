
library("optparse")

option_list = list(
    make_option(c("-i", "--input"), type="character", default="input_count.csv", help="input count file to be imputed [default= %default]", metavar="character"),
    make_option(c("-o", "--outputdir"), type="character", default="/output", help="directory which output should be stored in [default= %default]", metavar="character"),
    make_option(c("--gene_batch"), type="integer", default=50, help="number of genes per batch, therefore num_batches = choose_genes (or numgenes)/gene_batch. Max value is 150 [default= %default]", metavar="integer"),
    make_option(c("--num_iter"), type="integer", default=20, help="number of iterations, choose based on data size [default= %default]", metavar="integer"),
    make_option(c("--num_cells_batch"), type="integer", default=1000, help="set this to 1000 if input number of cells is in the 1000s, else set it to 100 [default= %default]", metavar="integer"),
    make_option(c("--alpha"), type="double", default=1.0, help="DPMM dispersion parameter. A higher value spins more clusters whereas a lower value spins lesser clusters [default= %default]", metavar="double"),
    make_option(c("--ncores"), type="integer", default=4, help="number of cores used in parallel computation [default= %default]", metavar="integer")
);


opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);


############## packages required ##############

library(MCMCpack)
library(mvtnorm)
library(ellipse)
library(coda)
library(Matrix)
library(Rtsne)
library(gtools)
library(foreach)
library(doParallel)
library(doSNOW)
library(snow)
library(lattice)
library(MASS)
library(bayesm)
library(robustbase)
library(chron)
library(mnormt)
library(schoolmath)
library(RColorBrewer)

#############################################


input_file_name <- "data.csv"
output_folder_name <- "output"
working_path <- "."

input_data_tab_delimited <- F
is_format_genes_cells <- TRUE
z_true_labels_avl <- F

gene_batch <- opt$gene_batch
num_iter <- opt$num_iter
num_cells_batch <- opt$num_cells_batch
alpha <- opt$alpha
num_cores <- opt$ncores

# choose_cells <- 3000; #comment if you want all the cells to be considered
# choose_genes <- 150; #comment if you want all the genes to be considered


# copy input file
file.copy(opt$input, "data.csv")

## call BISCUIT
source("BISCUIT_main.R")

# copy output directory
dir.create(opt$outputdir)
file.copy("output", opt$outputdir, recursive=TRUE)
