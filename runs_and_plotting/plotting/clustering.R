
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
  `no-impute`="no imputation",
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


dataset.mapping = list(
  clustering_baron_human="BARON-HUMAN",
  clustering_baron_mouse="BARON-MOUSE",
  clustering_cortex="CORTEX",
  clustering_pollen_lq="POLLEN-LQ",
  clustering_pollen_hq="POLLEN-HQ"
)

extract.bench.info = function(fname) {
  method = "NA"
  data.set = "NA"
  for (key in names(method.name.mapping)) {
    if (grepl(key, fname)) {
      method = method.name.mapping[[key]]
    }
  }
  for (ds in names(dataset.mapping)) {
    if (grepl(ds, fname)) {
      data.set = ds
    }
  }
  return(c(method, data.set))
}

if (!file.exists("./clustering.rds")) {
  
  
  
  test.results.dir = "/home/amirali/code/test\\ results"
  results = system(paste0("ls ", test.results.dir, "/*/clustering_*/result.txt"), intern = TRUE)
  results = c(results, system(paste0("ls ", "/home/amirali/code/cavatappi/results", "/*/clustering_*/result.txt"), intern = TRUE))
  

  
  extract.embedding.info = function(fname) {
    fname = tail(str_split(fname, "/")[[1]], n=1)
    fname = str_split(fname, "\\.")[[1]][1]
    for (alg in c("tsne", "pca", "ica", "umap", "truncated_svd")) {
      if (grepl(alg, fname)) {
        algorithm = alg
        class_name = str_remove(fname, paste0("_", alg))
      }
    }
    return(c(class_name, algorithm))
  }
  
  
  all.results = data.frame(data_set=character(), 
                           method=character(), 
                           criteria=character(), 
                           value=numeric(), 
                           stringsAsFactors=FALSE)
  
  all.embeddings = data.frame(data_set=character(), 
                              method=character(), 
                              algorithm=character(), 
                              class_name=character(),
                              X=numeric(),
                              Y=numeric(),
                              class=integer(),
                              k_means_clusters=integer(),
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
    
    
    embeddings = system(paste0("ls \"", bench.dir, "\"/files/*_*.csv"), intern = TRUE)
    for (embedding.filename in embeddings) {
      embedding.info = extract.embedding.info(embedding.filename)
      class_name = embedding.info[1]
      algorithm = embedding.info[2]
      embedding.data = read.csv(embedding.filename, stringsAsFactors = F)
      embedding.data = embedding.data[c("X", "Y", "class", "k_means_clusters")]
      embedding.data$data_set = data.set
      embedding.data$method = method
      embedding.data$algorithm = algorithm
      embedding.data$class_name = class_name
      
      all.embeddings = rbind(all.embeddings, embedding.data)
    }
  }
  
  saveRDS(list(all.results, all.embeddings), file = "./clustering.rds")
} else {
  data = readRDS(file = "./clustering.rds")
  all.results = data[[1]]
  all.embeddings = data[[2]]
}


all.results = all.results[all.results$method != "NA", ]
all.embeddings = all.embeddings[all.embeddings$method != "NA", ]

all.embeddings$random = runif(length(all.embeddings$method))

dataset.mapping = list(
  clustering_baron_human="BARON-HUMAN",
  clustering_baron_mouse="BARON-MOUSE",
  clustering_cortex="CORTEX",
  clustering_pollen_lq="POLLEN-LQ",
  clustering_pollen_hq="POLLEN-HQ"
)

dataset.size.mapping = list(
  clustering_baron_human=1,
  clustering_baron_mouse=1.8,
  clustering_cortex=1,
  clustering_pollen_lq=2.5,
  clustering_pollen_hq=2.5
)

dataset.alpha.mapping = list(
  clustering_baron_human=0.2,
  clustering_baron_mouse=0.5,
  clustering_cortex=0.5,
  clustering_pollen_lq=0.8,
  clustering_pollen_hq=0.8
)

algorithm.mapping = list(
  tsne="t-SNE", 
  pca="PCA",
  ica="ICA",
  umap="UMAP",
  truncated_svd="Truncated SVD"
)


criterion.2.score.mapping = function(criterion) {
  matched_score = "NA"
  for (score in c("calinski_harabaz", "silhouette", "v_measure", "adjusted_mutual_info")) {
    if (grepl(score, criterion)) {
      matched_score = score
    }
  }
  return(matched_score)
}

score.mapping = list(
  calinski_harabaz="Calinski-Harabaz score", 
  silhouette="Silhouette score",
  v_measure="V-Measure score (after k-means)",
  adjusted_mutual_info="Adjusted mutual info (after k-means)"
)

criterion.2.algorithm.mapping = function(criterion) {
  for (alg in c("tsne", "pca", "ica", "umap", "truncated_svd")) {
    if (grepl(alg, criterion)) {
      algorithm = alg
    }
  }
  return(algorithm)
}

criterion.2.class.mapping = function(criterion) {
  for (class in c("assigned_cluster", "class", "level1class", "level2class", "tissue")) {
    if (grepl(class, criterion)) {
      matched_class = class
    }
  }
  return(matched_class)
}


class.mapping = list(
  assigned_cluster="Class", 
  level1class="Level-1 Class",
  level2class="Level-2 Class",
  tissue="Tissue",
  class="Class"
)

all.results$score = sapply(all.results$criteria, function(x) {score.mapping[[criterion.2.score.mapping(x)]]})
all.results$class = sapply(all.results$criteria, function(x) {class.mapping[[criterion.2.class.mapping(x)]]})
all.results$algorithm = sapply(all.results$criteria, function(x) {algorithm.mapping[[criterion.2.algorithm.mapping(x)]]})







results=all.results
results = results[results$algorithm %in% c("PCA", "t-SNE", "UMAP"),]
results = results[results$score %in% c("Silhouette score", "V-Measure score (after k-means)"),]
results$criteria = NULL
results_casted = dcast(results, data_set + score + class + algorithm ~ method)

results_casted = results_casted[! results_casted$class %in% c("Tissue"),]
results_casted = results_casted[order(results_casted$score, results_casted$data_set, results_casted$class), ]

results_casted[-c(1, 2, 3, 4)] = round(results_casted[-c(1, 2, 3, 4)], 2)

results_casted_vm = results_casted[results_casted$score == "V-Measure score (after k-means)",]
results_casted_sl = results_casted[results_casted$score == "Silhouette score",]

results_casted_vm = results_casted_vm[, c(1, 2, 3, 4, order(
  - 1 * colSums(apply(results_casted_vm[-c(1, 2, 3, 4)], 1, FUN=max, na.rm=TRUE) == results_casted_vm[-c(1, 2, 3, 4)], na.rm = T) - 
    sapply(results_casted_vm[-c(1, 2, 3, 4)], FUN=mean, na.rm=TRUE)
) + 4)]

results_casted_sl = results_casted_sl[, c(1, 2, 3, 4, order(
  - 1 * colSums(apply(results_casted_sl[-c(1, 2, 3, 4)], 1, FUN=max, na.rm=TRUE) == results_casted_sl[-c(1, 2, 3, 4)], na.rm = T) - 
    sapply(results_casted_sl[-c(1, 2, 3, 4)], FUN=mean, na.rm=TRUE)
) + 4)]

print(xtable(t(results_casted_vm), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/clustering/v_measure.tex")
print(xtable(t(results_casted_sl), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/clustering/silhouette.tex")




levels = colnames(results_casted_sl)[colnames(results_casted_sl) != method.name.mapping[["no-impute"]]][-c(1)]
levels = levels[levels != "FNN"]
all.embeddings$method = factor(all.embeddings$method, levels=c("FNN", levels, method.name.mapping[["no-impute"]]))

for (data.set in unique(all.results$data_set)) {
  some.results = all.results[all.results$data_set == data.set,]
  some.embeddings = all.embeddings[all.embeddings$data_set == data.set,]
  for (class_name in unique(some.embeddings$class_name)) {
    for (algorithm in unique(some.embeddings$algorithm)) {
      custom.embeddings = some.embeddings[some.embeddings$algorithm == algorithm,]
      custom.embeddings = custom.embeddings[custom.embeddings$class_name == class_name,]
      
      embeddings.plot = ggplot(custom.embeddings, aes(x=X, y=Y, color=class, order=random)) + 
        geom_point(shape=16, size = dataset.size.mapping[[data.set]], alpha=dataset.alpha.mapping[[data.set]]) + 
        facet_wrap(~ method, scales="free", nrow=3) + theme_bw() +
        ggtitle(paste0("Result of performing ", 
                       algorithm.mapping[[algorithm]], " on ", 
                       dataset.mapping[[data.set]], " dataset", 
                       "after denoising by defferent methods")) +
        guides(colour=guide_legend(override.aes=list(alpha=1, size=4)))
      
      embeddings.plot = embeddings.plot + theme(axis.text.x = element_text(size=0), axis.text.y = element_text(size=0))
      
      if (length(unique(custom.embeddings$class <= 20))) {
        if (length(unique(custom.embeddings$class <= 10))) {
          embeddings.plot = embeddings.plot + scale_color_d3(palette = "category10")
        } else {
          embeddings.plot = embeddings.plot + scale_color_d3(palette = "category20")
        }
      } else {
        embeddings.plot = embeddings.plot + scale_color_igv()
      }
      
      
      pdf(paste0("./output/clustering/", class_name, ".", algorithm, ".on.", data.set, ".csv.scatter.plot.pdf"), width = 11, height = 7)
      print(embeddings.plot)
      dev.off()
    }
  }
}



plot.it = function(data.set, class_name, algorithm, save) {
  some.embeddings = all.embeddings[all.embeddings$data_set == data.set,]
  # some.embeddings$method = factor(some.embeddings$method, levels = colnames(results_casted_sl)[-c(1, 2, 3, 4)])
  custom.embeddings = some.embeddings[some.embeddings$algorithm == algorithm,]
  custom.embeddings = custom.embeddings[custom.embeddings$class_name == class_name,]
  
  embeddings.plot = ggplot(custom.embeddings, aes(x=X, y=Y, color=class, order=random)) + 
    geom_point(shape=16, size = dataset.size.mapping[[data.set]], alpha=dataset.alpha.mapping[[data.set]]) + 
    facet_wrap(~ method, scales="free", ncol = 5, strip.position = "top") + theme_bw() +
    theme(panel.grid = element_blank(), aspect.ratio = 1, axis.title = element_blank(),
          legend.position=c(0.77, 0.15),
          strip.background=element_blank(), plot.title = element_text(hjust = 0.5, size = 15),
          panel.spacing.y=unit(.3, "lines"), strip.text=element_text(size=15, margin=margin(2, 2, 2, 2, "mm"))) +
    # ggtitle(dataset.mapping[[data.set]]) +
    guides(colour=guide_legend(override.aes=list(alpha=1, size=4)))
  
  embeddings.plot = embeddings.plot + theme(axis.text.x = element_text(size=0), axis.text.y = element_text(size=0))
  print(length(unique(custom.embeddings$class)))
  
  if (length(unique(custom.embeddings$class)) <= 20) {
    embeddings.plot = embeddings.plot + scale_color_d3(palette = "category20")
    embeddings.plot = embeddings.plot + guides(colour=guide_legend(ncol=2, alpha=1, size=4))
  } else {
    embeddings.plot = embeddings.plot + scale_color_igv()
    embeddings.plot = embeddings.plot + guides(colour=guide_legend(ncol=5, alpha=1, size=4))
  }
  
  if (save) {
    # pdf(paste0("./output/clustering/scatter.", algorithm, ".on.", data.set, ".", class_name, ".plot.pdf"), width = 20, height = 30)
    pdf(paste0("./output/clustering/scatter.", algorithm, ".on.", data.set, ".", class_name, ".plot.pdf"), width = 10, height = 7)
    print(embeddings.plot)
    dev.off()
  } else {
    embeddings.plot
  }
  
}

plot.it("clustering_cortex", "level1class", "tsne", save=T)
plot.it("clustering_baron_human", "assigned_cluster", "tsne", save=T)
plot.it("clustering_baron_mouse", "assigned_cluster", "tsne", save=T)
plot.it("clustering_pollen_hq", "class", "tsne", save=T)
plot.it("clustering_pollen_lq", "class", "tsne", save=T)
plot.it("clustering_cortex", "level2class", "tsne", save=T)


plot.it("clustering_cortex", "level1class", "umap", save=T)
plot.it("clustering_baron_human", "assigned_cluster", "umap", save=T)
plot.it("clustering_baron_mouse", "assigned_cluster", "umap", save=T)
plot.it("clustering_pollen_hq", "class", "umap", save=T)
plot.it("clustering_pollen_lq", "class", "umap", save=T)
plot.it("clustering_cortex", "level2class", "umap", save=T)



for (data.set in unique(all.results$data_set)) {
  all.results.subset.1 = all.results[all.results$data_set == data.set,]
  for (class in unique(all.results.subset.1$class)) {
    all.results.subset.2 = all.results.subset.1[all.results.subset.1$class == class, ]
    for (algorithm in unique(all.results.subset.2$algorithm)) {
      all.results.subset.3 = all.results.subset.2[all.results.subset.2$algorithm == algorithm, ]
      for (score in unique(all.results.subset.3$score)) {
        some.results = all.results.subset.3[all.results.subset.3$score == score, ]
        
        results.plot = ggplot(some.results, aes(x=reorder(reorder(method, -value), method!="Ours"), y=value)) + 
          geom_bar(stat="identity", position=position_dodge(), fill="red") +
          theme_bw() + scale_fill_brewer(palette="Set1") +
          theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
          ggtitle(paste0(score, " on data", 
                         " after dimension reduction by ", algorithm, ".",
                         "\n", dataset.mapping[[data.set]], " dataset.",
                         " Colors corrspond to ", tolower(class), " values.")) + 
          theme(plot.title = element_text(size = 12)) +
          ylab(score) + 
          xlab("")
        
        pdf(paste0("./output/clustering/metric.", data.set, 
                   ".", class, ".", algorithm, ".", score, ".plot.pdf"), width = 8, height = 5)
        print(results.plot)
        dev.off()
      }
    }
  }
}

for (data.set in unique(all.results$data_set)) {
  all.results.subset.1 = all.results[all.results$data_set == data.set,]
  for (class in unique(all.results.subset.1$class)) {
    all.results.subset.2 = all.results.subset.1[all.results.subset.1$class == class, ]
    for (algorithm in unique(all.results.subset.2$algorithm)) {
      some.results = all.results.subset.2[all.results.subset.2$algorithm == algorithm, ]
      
      some.results[some.results$score == "Calinski-Harabaz score",]$value = 
        some.results[some.results$score == "Calinski-Harabaz score",]$value / 
        max(some.results[some.results$score == "Calinski-Harabaz score",]$value)
      some.results[some.results$score == "Calinski-Harabaz score",]$score = "Scaled Calinski-Harabaz score"
      results.plot = ggplot(some.results, aes(x=reorder(reorder(method, -value), method!="Ours"), y=value, fill=score)) + 
        geom_bar(stat="identity", position=position_dodge()) +
        theme_bw() + scale_fill_brewer(palette="Set1") +
        theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
        ggtitle(paste0("Benchmarking ",
                       " after dimension reduction by ", algorithm, ".",
                       "\n",  dataset.mapping[[data.set]], " dataset. ", 
                       "Colors corrspond to ", tolower(class), " values.")) + 
        theme(plot.title = element_text(size = 12)) +
        ylab("") + 
        xlab("")
      
      pdf(paste0("./output/clustering/multi.metric.", data.set, 
                 ".", class, ".", algorithm, ".plot.pdf"), width = 8, height = 5)
      print(results.plot)
      dev.off()
    }
  }
}
