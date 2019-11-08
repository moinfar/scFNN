
library(stringr)
library(ggplot2)

library(reshape2)
library(xtable)




method.name.mapping = list(
  dca="DCA",
  deepimpute="DeepImpute",
  drimpute="DrImpute",
  `knn-smoothing`="KNN-smoothing",
  magic="MAGIC",
  `no-impute`="no imputation",
  saucie="SAUCIE",
  saver="SAVER",
  scimpute="scImpute",
  scscope="scScope",
  scvi="scVI",
  uncurl="UNCURL",
  zinbwave="ZINB-WaVE",
  # FFL_ZINB="FFL-ZINB",
  FFL_old_ZINB="FNN"
)


if (!file.exists("./cite_seq.rds")) {
  
  
  test.results.dir = "/home/amirali/code/test\\ results"
  results = system(paste0("ls ", test.results.dir, "/*/cite_seq*/result.txt"), intern = TRUE)
  results = c(results, system(paste0("ls ", "/home/amirali/code/cavatappi/results", "/*/cite_seq*/result.txt"), intern = TRUE))
  
  
  extract.bench.info = function(fname) {
    method = "NA"
    data.set = "NA"
    for (key in names(method.name.mapping)) {
      if (grepl(key, fname)) {
        method = method.name.mapping[[key]]
      }
    }
    for (ds in c("cite_seq_cbmc", "cite_seq_cd8", "cite_seq_pbmc")) {
      if (grepl(ds, fname)) {
        data.set = ds
      }
    }
    return(c(method, data.set))
  }
  
  all.results = data.frame(data_set=character(), 
                           method=character(), 
                           criteria=character(), 
                           value=numeric(), 
                           stringsAsFactors=FALSE)
  
  for (filename in results) {
    print(filename)
    bench.dir = dirname(filename)
    
    bench.info = extract.bench.info(filename)
    method = bench.info[1]
    data.set = bench.info[2]
    
    result.file = read.csv(filename, comment.char = "#", sep = "\t", stringsAsFactors = F,
                           col.names = c("criteria", "value"), header=F)
    result.file$method = method
    result.file$data_set = data.set
    
    all.results = rbind(all.results, result.file)
  }
  
  
  saveRDS(list(all.results), file = "./cite_seq.rds")
} else {
  data = readRDS(file = "./cite_seq.rds")
  all.results = data[[1]]
}

all.results = all.results[all.results$method != "NA", ]



dataset.mapping = list(
  cite_seq_cbmc="CITE-CBMC",
  cite_seq_cd8="CITE-CD8",
  cite_seq_pbmc="CITE-PBMC"
)

criteria.mapping = list(
  MSE_of_adt_adt_and_rna_rna_pearson_correlations="MSE between RNA/RNA Pearson correlations and protein/protein correlations",
  MSE_of_adt_adt_and_rna_rna_spearman_correlations="MSE between RNA/RNA Spearman correlations and protein/protein correlations",
  rna_protein_mean_pearson_correlatoin="Average of Pearson correlations between RNAs and corresponding proteins",
  rna_protein_mean_spearman_correlatoin="Average of Spearman correlations between RNAs and corresponding proteins"
)

all.results$data_set = sapply(all.results$data_set, function(x) {dataset.mapping[[x]]})






results=all.results
results = results[results$criteria %in% c("rna_protein_mean_pearson_correlatoin", "rna_protein_mean_spearman_correlatoin"),]
results_casted = dcast(results, data_set + criteria ~ method)

results_casted = results_casted[order(results_casted$criteria, results_casted$data_set), ]

results_casted = results_casted[, c(1, 2, order(
  - 1 * colSums(apply(results_casted[-c(1, 2)], 1, FUN=max, na.rm=TRUE) == results_casted[-c(1, 2)], na.rm = T) - 
    sapply(results_casted[-c(1, 2)], FUN=mean, na.rm=TRUE)
) + 2)]

results_casted[-c(1, 2)] = round(results_casted[-c(1, 2)], 2)

print(xtable(t(results_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/cite_seq/results.tex")
















for (criterion in unique(all.results$criteria)) {
  some.results = all.results[all.results$criteria == criterion,]
  if (grepl("MSE", criterion)) {
    results.plot = ggplot(some.results, aes(x=reorder(method, value), y=value, fill=data_set))
  } else {
    results.plot = ggplot(some.results, aes(x=reorder(method, -value), y=value, fill=data_set))
  }
  results.plot = results.plot + geom_bar(stat="identity", position=position_dodge()) +
    theme_bw() + scale_fill_brewer(palette="Set1") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
    ggtitle(paste0(criteria.mapping[[criterion]], "\nColors correspond to different datasets")) + 
    theme(plot.title = element_text(size = 12)) +
    ylab("") + 
    xlab("")
  
  pdf(paste0("./output/cite_seq/all.", criterion, ".plot.pdf"), width = 8, height = 5)
  print(results.plot)
  dev.off()
}


for (criterion in unique(all.results$criteria)) {
  for (data.set in unique(all.results$data_set)) {
    some.results = all.results[all.results$criteria == criterion,]
    some.results = some.results[some.results$data_set == data.set,]
    if (grepl("MSE", criterion)) {
      results.plot = ggplot(some.results, aes(x=reorder(method, value), y=value))
    } else {
      results.plot = ggplot(some.results, aes(x=reorder(method, -value), y=value))
    }
    results.plot = results.plot + geom_bar(stat="identity", position=position_dodge(), fill="red") +
      theme_bw() + scale_fill_brewer(palette="Set1") +
      theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
      ggtitle(paste0(criteria.mapping[[criterion]], " on ", data.set, " dataset.")) + 
      theme(plot.title = element_text(size = 12)) +
      ylab("") + 
      xlab("")
    
    pdf(paste0("./output/cite_seq/", data.set, ".", criterion, ".plot.pdf"), width = 8, height = 5)
    print(results.plot)
    dev.off()
  }
}

