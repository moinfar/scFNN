library(optparse)
library(SAVER)

packageVersion("SAVER")

option_list = list(
    make_option(c("-i", "--input"), type="character", default="input_count.csv", help="input count file to be imputed [default= %default]", metavar="character"),
    make_option(c("-o", "--outputdir"), type="character", default="/output", help="directory which output should be stored in [default= %default]", metavar="character"),
    make_option(c("--ncores"), type="integer", default=4, help="number of cores used in parallel computation [default= %default]", metavar="integer")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if(!file.exists(opt$input)) {
    stop(paste0("File `", input_file, "` not exists. Did you mount the containing volume into docker?"))
}

X = read.csv(opt$input, header = TRUE, row.names = 1)
X = as.matrix(X)
dim(X)

X.saver <- saver(X, ncores = opt$ncores)

dir.create(opt$outputdir)

write.csv(X.saver$estimate, file.path(opt$outputdir, "saver_estimates.csv"))
write.csv(X.saver$se, file.path(opt$outputdir, "saver_se.csv"))
saveRDS(X.saver$info, file.path(opt$outputdir, "info.rds"))
