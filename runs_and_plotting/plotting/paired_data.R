
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



if (!file.exists("./paired_data.rds")) {
  
  
  test.results.dir = "/home/amirali/code/test\\ results"
  results = system(paste0("ls ", test.results.dir, "/*/paired_data*/result.txt"), intern = TRUE)
  results = c(results, system(paste0("ls ", "/home/amirali/code/cavatappi/results", "/*/paired_data*/result.txt"), intern = TRUE))
  
  
  extract.bench.info = function(fname) {
    method = "NA"
    for (key in names(method.name.mapping)) {
      if (grepl(key, fname)) {
        method = method.name.mapping[[key]]
      }
    }
    return(method)
  }
  
  all.results = data.frame(method=character(), 
                           criteria=character(), 
                           value=numeric(), 
                           stringsAsFactors=FALSE)
  
  for (filename in results) {
    print(filename)
    bench.dir = dirname(filename)
    
    method = extract.bench.info(filename)
    
    result.file = read.csv(filename, comment.char = "#", sep = "\t", stringsAsFactors = F,
                           col.names = c("criteria", "value"), header=F)
    result.file$method = method
    all.results = rbind(all.results, result.file)
  }
  
  
  saveRDS(list(all.results), file = "./paired_data.rds")
} else {
  data = readRDS(file = "./paired_data.rds")
  all.results = data[[1]]
}

all.results = all.results[all.results$method != "NA", ]



all.results$value[is.na(all.results$value)] = 1

criteria.mapping = list(
  cell_mean_absolute_error="MAE",
  cell_mean_correlation_distance="Mean correlation distance",
  cell_mean_cosine_distance="Mean cosine distance",
  cell_mean_euclidean_distance="Mean euclidean distance",
  cell_root_mean_squared_error="RMSE"
  
)





results=all.results
results_casted = dcast(results, criteria ~ method)
results_casted = results_casted[, c(1, order(
  - 10 * colSums(apply(results_casted[-c(1)], 1, FUN=min, na.rm=TRUE) == results_casted[-c(1)], na.rm = T) + 
    sapply(results_casted[-c(1)], FUN=median, na.rm=TRUE)
) + 1)]
results_casted[-c(1)] = round(results_casted[-c(1)], 2)
results_casted = results_casted[c(5,1,3), ]
print(xtable(t(results_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/paired_data/results.tex")










for (criterion in unique(all.results$criteria)) {
  some.results = all.results[all.results$criteria == criterion,]
  
  results.plot = ggplot(some.results, aes(x=reorder(method, value), y=value)) +
    geom_bar(stat="identity", position=position_dodge(), fill="red") +
    theme_bw() + scale_fill_brewer(palette="Set1") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
    ggtitle(paste0(criteria.mapping[[criterion]], " between the results and Pollen HQ data",
                   "\n Only entries which are zero in LQ data and nonzero in HQ data are considered",
                   "\nAveraging is performed over cells")) + 
    theme(plot.title = element_text(size = 12)) +
    ylab("") + 
    xlab("")
  
  pdf(paste0("./output/paired_data/", criterion, ".plot.pdf"), width = 8, height = 5)
  print(results.plot)
  dev.off()
}

some.results = all.results[all.results$criteria != "cell_mean_euclidean_distance",]
some.results$criteria = sapply(some.results$criteria, function(x) {criteria.mapping[[x]]})

results.plot = ggplot(some.results, aes(x=reorder(method, value), y=value, fill=criteria)) +
  geom_bar(stat="identity", position=position_dodge()) +
  theme_bw() + scale_fill_brewer(palette="Set1") +
  theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
  ggtitle(paste0("Distances between the results and Pollen HQ data",
                 "\n Only entries which are zero in LQ data and nonzero in HQ data are considered",
                 "\nAveraging is performed over cells")) + 
  theme(plot.title = element_text(size = 12)) +
  ylab("") + 
  xlab("")

pdf(paste0("./output/paired_data/all", ".plot.pdf"), width = 8, height = 5)
print(results.plot)
dev.off()
