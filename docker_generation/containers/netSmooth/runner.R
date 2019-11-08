library("optparse")

option_list = list(
    make_option(c("-i", "--input"), type="character", default="input_count.csv", help="input count file to be imputed [default= %default]", metavar="character"),
    make_option(c("-o", "--outputdir"), type="character", default="/output", help="directory which output should be stored in [default= %default]", metavar="character"),
    make_option(c("--organism"), type="character", default="human", help="Organism which PPI network will be used for (choices: smallhuman/human/mouse) [default=human]", metavar="ORGN"),
    make_option(c("--alpha"), type="double", default=-1, help="Alpha in the algorithm (enter -1 for auto) [default=-1]", metavar="double")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if(!file.exists(opt$input)) {
    stop(paste0("File `", input_file, "` not exists. Did you mount the containing volume into docker?"))
}

alpha = "auto"
if (opt$alpha > 0) {
  alpha = opt$alpha
}

library(netSmooth)

if (opt$organism == "smallhuman") {
  data(smallPPI)
  PPI = smallPPI
}
if (opt$organism == "human") {
  data(human.ppi)
  PPI = human.ppi
}
if (opt$organism == "mouse") {
  data(mouse.ppi)
  PPI = mouse.ppi
}

X = read.csv(opt$input, header = TRUE, row.names = 1)
X = as.matrix(X)

SE = SummarizedExperiment(assays=list(counts=X))

smoothed.SE <- netSmooth(SE, PPI, alpha=alpha)


dir.create(opt$outputdir)
write.csv(smoothed.SE, file.path(opt$outputdir, "netSmooth_output.csv"))

print("Done")
