library("optparse")

option_list = list(
    make_option(c("-i", "--input"), type="character", default="input_count.csv", help="input count file to be imputed [default= %default]", metavar="character"),
    make_option(c("-o", "--outputdir"), type="character", default="/output", help="directory which output should be stored in [default= %default]", metavar="character")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if(!file.exists(opt$input)) {
    stop(paste0("File `", input_file, "` not exists. Did you mount the containing volume into docker?"))
}

library(DrImpute)

X = read.csv(opt$input, header = TRUE, row.names = 1)

X.log <- log(X + 1)
X.imp <- DrImpute(X.log)

dir.create(opt$outputdir)
write.csv(X.imp, file.path(opt$outputdir, "drimpute_output.csv"))
