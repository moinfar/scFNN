library("optparse")

option_list = list(
    make_option(c("-i", "--input"), type="character", default="input_count.csv", help="input count file to be imputed [default= %default]", metavar="character"),
    make_option(c("-o", "--outputdir"), type="character", default="/output", help="directory which output should be stored in [default= %default]", metavar="character"),
    make_option(c("--drop_thre"), type="double", default=0.5, help="threshold set on dropout probability [default= %default]", metavar="double"),
    make_option(c("--Kcluster"), type="integer", default=2, help="number of cell subpopulations for the algorithm [default= %default]", metavar="integer"),
    make_option(c("--ncores"), type="integer", default=4, help="number of cores used in parallel computation [default= %default]", metavar="integer")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if(!file.exists(opt$input)) {
    stop(paste0("File `", input_file, "` not exists. Did you mount the containing volume into docker?"))
}

library(scImpute)

scimpute(count_path = opt$input,
         infile = "csv",
         outfile = "csv",
         out_dir = opt$outputdir,
         labeled = FALSE,
         drop_thre = opt$drop_thre,
         Kcluster = opt$Kcluster,
         ncores = opt$ncores)
