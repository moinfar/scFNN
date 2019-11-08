library("optparse")

option_list = list(
    make_option(c("-i", "--input"), type="character", default="input_count.csv", help="input count file to be imputed [default= %default]", metavar="character"),
    make_option(c("-o", "--outputdir"), type="character", default="/output", help="directory which output should be stored in [default= %default]", metavar="character"),
    make_option(c("-e", "--epsilon"), type="double", default=1e12, help="epsilon [default= %default]", metavar="double"),
    make_option(c("-k", "--k"), type="integer", default=2, help="K [default= %default]", metavar="integer"),
    make_option(c("-g", "--gene_subset"), type="integer", default=0, help="Number of HVGs to be analysed (enter 0 for all) [default= %default]", metavar="integer"),
    make_option(c("--ncores"), type="integer", default=4, help="number of cores used in parallel computation [default= %default]", metavar="integer")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

if(!file.exists(opt$input)) {
    stop(paste0("File `", input_file, "` not exists. Did you mount the containing volume into docker?"))
}


library(doParallel)
library(BiocParallel)
NCORES = opt$ncores
registerDoParallel(NCORES)
register(DoparParam())


library(SummarizedExperiment)
library(zinbwave)
library(magrittr)

X = read.csv(opt$input, header = TRUE, row.names = 1)
X = as.matrix(X)

SE = SummarizedExperiment(assays=list(counts=X))

if (opt$gene_subset > 0) {
  assay(SE) %>% log1p %>% rowVars -> vars
  names(vars) <- rownames(SE)
  vars <- sort(vars, decreasing = TRUE)
  hvgs = names(vars)[1:opt$gene_subset]
  SE <- SE[hvgs, ]
}

SE_zinb <- zinbwave(SE, K=opt$k, epsilon=opt$epsilon,
                    imputedValues=T, normalizedValues=T)

emb <- reducedDim(SE_zinb)
weights = assay(SE_zinb, "weights")
imputedValues = assay(SE_zinb, "imputedValues")
normalizedValues = assay(SE_zinb, "normalizedValues")

if (opt$gene_subset > 0) {
  full_imputed_values = X
  full_imputed_values[hvgs, ] = imputedValues
}

dir.create(opt$outputdir)
write.csv(emb, file.path(opt$outputdir, "embedding.csv"))
write.csv(weights, file.path(opt$outputdir, "weights.csv"))
write.csv(imputedValues, file.path(opt$outputdir, "imputed_values.csv"))
write.csv(normalizedValues, file.path(opt$outputdir, "normalized_values.csv"))

if (opt$gene_subset > 0) {
  write.csv(full_imputed_values, file.path(opt$outputdir, "full_imputed_values.csv"))
}

print("Done!")
