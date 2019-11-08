
library(stringr)
library(ggplot2)
library(ggsci)
library(reshape2)
library(xtable)



method.name.mapping = list(
  dca="DCA",
  deepimpute="DeepImpute",
  drimpute="DrImpute",
  `knn-smoothing`="KNN-smoothing",
  magic="MAGIC",
  `no-impute`="Not Imputed",
  saucie="SAUCIE",
  saver="SAVER",
  scimpute="scImpute",
  # scscope="scScope",
  scvi="scVI",
  uncurl="UNCURL",
  zinbwave="ZINB-WaVE",
  # FFL_ZINB="FFL-ZINB",
  FFL_old_ZINB="FNN"
)



if (!file.exists("./cell_cycle.rds")) {
  
  test.results.dir = "/home/amirali/code/test\\ results"
  results = system(paste0("ls ", test.results.dir, "/*/cell_cycle_*/result.txt"), intern = TRUE)
  results = c(results, system(paste0("ls ", "/home/amirali/code/cavatappi/results", "/*/cell_cycle_*/result.txt"), intern = TRUE))
  
  
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
  
  
  all.embeddings = data.frame(method=character(), 
                              algorithm=character(), 
                              X=numeric(),
                              Y=numeric(),
                              class=integer(),
                              k_means_clusters=integer(),
                              stringsAsFactors=FALSE)
  
  
  for (filename in results) {
    print(filename)
    bench.dir = dirname(filename)
    
    method = extract.bench.info(filename)
    
    result.file = read.csv(filename, comment.char = "#", sep = "\t", stringsAsFactors = F,
                           col.names = c("criteria", "value"), header=F)
    result.file$method = method
    
    all.results = rbind(all.results, result.file)
    
    for (alg in c("tsne", "pca", "ica", "umap", "truncated_svd")) {
      embedding.data = read.csv(paste0(bench.dir, "/files/", alg, ".csv"), stringsAsFactors = F)
      embedding.data = embedding.data[c("X", "Y", "class", "k_means_clusters")]
      embedding.data$method = method
      embedding.data$algorithm = alg
      all.embeddings = rbind(all.embeddings, embedding.data)
    }
  }
  
  saveRDS(list(all.results, all.embeddings), file = "./cell_cycle.rds")
} else {
  data = readRDS(file = "./cell_cycle.rds")
  all.results = data[[1]]
  all.embeddings = data[[2]]
}


all.results = all.results[all.results$method != "NA", ]
all.embeddings = all.embeddings[all.embeddings$method != "NA", ]


algorithm.mapping = list(
  tsne="t-SNE", 
  pca="PCA",
  ica="ICA",
  umap="UMAP",
  truncated_svd="Truncated SVD",
  knn="KNN",
  svm="SVM",
  identity="no dimension reduction",
  related_genes="no dimension reduction"
)




results=all.results
results_casted = dcast(results, criteria ~ method)
results_casted = results_casted[results_casted$criteria %in% c("classification_knn_mean_accuracy", "classification_svm_mean_accuracy", "embedding_identity_silhouette_score"),]
results_casted = results_casted[, c(1, order(
  - 10 * colSums(apply(results_casted[-c(1)], 1, FUN=max, na.rm=TRUE) == results_casted[-c(1)], na.rm = T) - 
    sapply(results_casted[-c(1)], FUN=mean, na.rm=TRUE)
) + 1)]
results_casted[-c(1)] = round(results_casted[-c(1)], 2)
results_casted = results_casted[c(3,1,2), ]
print(xtable(t(results_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/cell_cycle/results.tex")



all.embeddings$random = runif(length(all.embeddings$method))
levels = colnames(results_casted)[colnames(results_casted) != method.name.mapping[["no-impute"]]][-c(1)]
all.embeddings$method = factor(all.embeddings$method, levels=c(levels, method.name.mapping[["no-impute"]]))

for (algorithm in unique(all.embeddings$algorithm)) {
  custom.embeddings = all.embeddings[all.embeddings$algorithm == algorithm,]
  
  accuracy <- data.frame(method=colnames(results_casted[,-1]), accuracy=unlist(results_casted[2,-1]))
  
  embeddings.plot = ggplot(custom.embeddings) + 
    geom_point(aes(x=X, y=Y, color=class, order=random), shape=16, size = .8, alpha=1) + scale_color_aaas() +
    facet_wrap(~ method, scales="free", nrow=3) + theme_bw() +
    # ggtitle(paste0("Result of performing ", 
    # algorithm.mapping[[algorithm]], " on marker genes of CELL-CYCLE", 
    # " after denoising by defferent methods")) +
    xlab(paste0(algorithm.mapping[[algorithm]], " 1")) + ylab(paste0(algorithm.mapping[[algorithm]], " 2")) + 
    labs(color = "Cell Stage") +
    guides(colour=guide_legend(override.aes=list(alpha=1, size=5))) +
    theme(panel.grid = element_blank(), axis.ticks=element_blank(),
          axis.text.x = element_text(size=0), axis.text.y = element_text(size=0),
          strip.background=element_blank(), legend.position=c(0.91, 0.13),
          strip.text=element_text(size=9, margin=margin(0, 0, 1, 0, "mm"))) +geom_text(data=accuracy, aes(x=Inf, y=Inf, label=accuracy), vjust=1.2, hjust=1.1)
  
  
  pdf(paste0("./output/cell_cycle/", algorithm.mapping[[algorithm]], ".scatter.plot.pdf"), width = 7.3, height = 5)
  print(embeddings.plot)
  dev.off()
}





criterion.2.score.mapping = function(criterion) {
  for (score in c("calinski_harabaz", "silhouette", "v_measure", "adjusted_mutual_info", "accuracy")) {
    if (grepl(score, criterion)) {
      matched_score = score
    }
  }
  return(matched_score)
}

score.mapping = list(
  calinski_harabaz="Calinski-Harabaz score", 
  silhouette="Silhouette score",
  v_measure="V-Measure score",
  adjusted_mutual_info="Adjusted mutual info (after k-means)",
  accuracy="Accuracy"
)

criterion.2.algorithm.mapping = function(criterion) {
  for (alg in c("tsne", "pca", "ica", "umap", "truncated_svd", "svm", "knn", "related_genes", "identity")) {
    if (grepl(alg, criterion)) {
      algorithm = alg
    }
  }
  return(algorithm)
}


all.results$score = sapply(all.results$criteria, function(x) {score.mapping[[criterion.2.score.mapping(x)]]})
all.results$algorithm = sapply(all.results$criteria, function(x) {algorithm.mapping[[criterion.2.algorithm.mapping(x)]]})



for (score in unique(all.results$score)) {
  all.results.subset.1 = all.results[all.results$score == score, ]
  for (algorithm in unique(all.results.subset.1$algorithm)) {
    some.results = all.results.subset.1[all.results.subset.1$algorithm == algorithm, ]
    
    results.plot = ggplot(some.results, aes(x=reorder(method, -value), y=value)) + 
      geom_bar(stat="identity", position=position_dodge(), fill="red") +
      theme_bw() + scale_fill_brewer(palette="Set1") +
      theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
      ggtitle(paste0(score, " on genes related to cell-cycle", 
                     " after dimension reduction by ", algorithm, ".")) + 
      theme(plot.title = element_text(size = 12)) +
      ylab("") + 
      xlab("")
    
    pdf(paste0("./output/cell_cycle/", algorithm, 
               ".", score, ".plot.pdf"), width = 8, height = 5)
    print(results.plot)
    dev.off()
  }
}


for (score in unique(all.results$score)) {
  all.results.subset.1 = all.results[all.results$score == score, ]
  some.results = all.results.subset.1
  
  results.plot = ggplot(some.results, aes(x=reorder(method, -value), y=value, fill=algorithm)) + 
    geom_bar(stat="identity", position=position_dodge()) +
    theme_bw() + scale_fill_brewer(palette="Set1") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
    # ggtitle(paste0(score, " on genes related to cell-cycle", 
    # " after dimension reduction. ", 
    # "\nColors corrspond to different dimension reduction algorithms")) + 
    theme(plot.title = element_text(size = 12), legend.text = element_text(size = 12),
          legend.position=c(0.88, 0.88), legend.background = element_blank()) +
    labs(fill="Classification Algorithm") +
    ylab("Accuracy") + 
    xlab("")
  
  pdf(paste0("./output/cell_cycle/", "all", 
             ".", score, ".plot.pdf"), width = 8, height = 5)
  print(results.plot)
  dev.off()
}

for (algorithm in unique(all.results$algorithm)) {
  all.results.subset.1 = all.results[all.results$algorithm == algorithm, ]
  some.results = all.results.subset.1
  
  some.results[some.results$score == "Calinski-Harabaz score",]$value = 
    some.results[some.results$score == "Calinski-Harabaz score",]$value / 
    max(some.results[some.results$score == "Calinski-Harabaz score",]$value)
  
  results.plot = ggplot(some.results, aes(x=reorder(method, -value), y=value, fill=score)) + 
    geom_bar(stat="identity", position=position_dodge()) +
    theme_bw() + scale_fill_brewer(palette="Set1") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
    ggtitle(paste0("Performance measures on genes related to cell-cycle", 
                   " after ", algorithm, ".", 
                   "\nColors corrspond to different dimension reduction algorithms")) + 
    theme(plot.title = element_text(size = 12)) +
    ylab("") + 
    xlab("")
  
  pdf(paste0("./output/cell_cycle/", algorithm, 
             ".", "all", ".plot.pdf"), width = 8, height = 5)
  print(results.plot)
  dev.off()
}



results = all.results[(all.results$algorithm == "no dimension reduction" & all.results$score == "Silhouette score")
                      | all.results$algorithm == "KNN" | all.results$algorithm == "SVM",]

results$algorithm = paste0(results$algorithm, " ")
results$algorithm[results$algorithm == "no dimension reduction "] = ""
results$score = paste0(results$algorithm, "", results$score)


results.plot = ggplot(results, aes(x=reorder(method, -value), y=value, fill=reorder(score, value) )) + 
  geom_bar(stat="identity", position=position_dodge()) +
  theme_bw() + scale_fill_npg() +
  theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
  ggtitle(paste0(score, " on genes related to cell-cycle", 
                 " after dimension reduction. ", 
                 "\nColors corrspond to different dimension reduction algorithms")) + 
  theme(plot.title = element_text(size = 12)) +
  ylab("") + 
  xlab("")

pdf(paste0("./output/cell_cycle/", "all.paper.plot.pdf"), width = 8, height = 5)
print(results.plot)
dev.off()


