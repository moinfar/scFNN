
library(stringr)
library(ggplot2)
library(reshape2)
library(xtable)
library(ggsci)
library(gridExtra)



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
  random_mask_10xPBMC4k_1="10x-PBMC4k",
  random_mask_10xPBMC4k_2="10x-PBMC4k",
  random_mask_baron_human_1="BARON-HUMAN",
  random_mask_baron_human_2="BARON-HUMAN",
  random_mask_baron_mouse_1="BARON-MOUSE",
  random_mask_baron_mouse_2="BARON-MOUSE",
  random_mask_cell_cycle_1="CELL-CYCLE",
  random_mask_cell_cycle_2="CELL-CYCLE",
  random_mask_cite_cd8_1="CITE-CD8",
  random_mask_cite_cd8_2="CITE-CD8",
  random_mask_cortex_1="CORTEX",
  random_mask_cortex_2="CORTEX",
  random_mask_pollen_hq_1="POLLEN-HQ",
  random_mask_pollen_hq_2="POLLEN-HQ",
  random_mask_pollen_lq_1="POLLEN-LQ",
  random_mask_pollen_lq_2="POLLEN-LQ"
)




if (!file.exists("./random_mask.rds")) {
  test.results.dir = "/home/amirali/code/test\\ results"
  results = system(paste0("ls ", test.results.dir, "/*/random_mask_*/result.txt"), intern = TRUE)
  results = c(results, system(paste0("ls ", "/home/amirali/code/cavatappi/results", "/*/random_mask_*/result.txt"), intern = TRUE))
  
  
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
  
  all.results = data.frame(data_set=character(), 
                           method=character(), 
                           criteria=character(), 
                           value=numeric(), 
                           stringsAsFactors=FALSE)
  
  all.predictions = data.frame(data_set=character(), 
                               method=character(), 
                               original=numeric(), 
                               pedicted=numeric(), 
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
    
    predictions = read.csv(paste0(bench.dir, "/files/predictions.csv"), stringsAsFactors = F)
    predictions = predictions[,2:3]
    predictions[predictions < 0] = 0
    predictions$method = method
    predictions$data_set = data.set
    
    all.predictions = rbind(all.predictions, predictions)
  }
  
  saveRDS(list(all.results, all.predictions), file = "./random_mask.rds")
} else {
  data = readRDS(file = "./random_mask.rds")
  all.results = data[[1]]
  all.predictions = data[[2]]
}

all.predictions$predicted[is.na(all.predictions$predicted)] = 0



all.results = all.results[all.results$method !=  "NA", ]
all.predictions = all.predictions[all.predictions$method != "NA", ]


title.mapping = list(
  random_mask_10xPBMC4k_1="Predictions vs. original values (log scaled)\n10,000 entries of 10x-PBMC4k dataset with minimum expression of 10 UMIs are dropped.",
  random_mask_10xPBMC4k_2="Predictions vs. original values (log scaled)\n10,000 entries of 10x-PBMC4k dataset from top %10 HGV genes with minimum expression of 10 UMIs are dropped.",
  random_mask_baron_human_1="Predictions vs. original values (log scaled)\n10,000 entries of BARON-HUMAN dataset with minimum expression of 10 counts are dropped.",
  random_mask_baron_human_2="Predictions vs. original values (log scaled)\n10,000 entries of BARON-HUMAN dataset from top %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_baron_mouse_1="Predictions vs. original values (log scaled)\n10,000 entries of BARON-MOUSE dataset with minimum expression of 10 counts are dropped.",
  random_mask_baron_mouse_2="Predictions vs. original values (log scaled)\n10,000 entries of BARON-MOUSE dataset from top %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_cell_cycle_1="Predictions vs. original values (log scaled)\n10,000 entries of CELL-CYCLE dataset with minimum expression of 10 counts are dropped.",
  random_mask_cell_cycle_2="Predictions vs. original values (log scaled)\n10,000 entries of CELL-CYCLE dataset from top %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_cite_cd8_1="Predictions vs. original values (log scaled)\n1,000 entries of CITE-CD8 dataset with minimum expression of 10 UMIs are dropped.",
  random_mask_cite_cd8_2="Predictions vs. original values (log scaled)\n1,000 entries of CITE-CD8 dataset from top %10 HGV genes with minimum expression of 10 UMIs are dropped.",
  random_mask_cortex_1="Predictions vs. original values (log scaled)\n10,000 entries of CORTEX dataset with minimum expression of 10 UMIs are dropped.",
  random_mask_cortex_2="Predictions vs. original values (log scaled)\n10,000 entries of CORTEX dataset from top %10 HGV genes with minimum expression of 10 UMIs are dropped.",
  random_mask_pollen_hq_1="Predictions vs. original values (log scaled)\n10,000 entries of POLLEN-HQ dataset with minimum expression of 10 counts are dropped.",
  random_mask_pollen_hq_2="Predictions vs. original values (log scaled)\n10,000 entries of POLLEN-HQ dataset from top %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_pollen_lq_1="Predictions vs. original values (log scaled)\n10,000 entries of POLLEN-LQ dataset with minimum expression of 10 counts are dropped.",
  random_mask_pollen_lq_2="Predictions vs. original values (log scaled)\n10,000 entries of POLLEN-LQ dataset from top %10 HGV genes with minimum expression of 10 counts are dropped."
)


title.mapping.2 = list(
  random_mask_10xPBMC4k_1="Prediction error on dropped values (log scaled)\n10,000 entries of 10x-PBMC4k dataset with \nminimum expression of 10 UMIs are dropped.",
  random_mask_10xPBMC4k_2="Prediction error on dropped values (log scaled)\n10,000 entries of 10x-PBMC4k dataset from \ntop %10 HGV genes with minimum expression of 10 UMIs are dropped.",
  random_mask_baron_human_1="Prediction error on dropped values (log scaled)\n10,000 entries of BARON-HUMAN dataset with \nminimum expression of 10 counts are dropped.",
  random_mask_baron_human_2="Prediction error on dropped values (log scaled)\n10,000 entries of BARON-HUMAN dataset from \ntop %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_baron_mouse_1="Prediction error on dropped values (log scaled)\n10,000 entries of BARON-MOUSE dataset with \nminimum expression of 10 counts are dropped.",
  random_mask_baron_mouse_2="Prediction error on dropped values (log scaled)\n10,000 entries of BARON-MOUSE dataset from \ntop %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_cell_cycle_1="Prediction error on dropped values (log scaled)\n10,000 entries of CELL-CYCLE dataset with \nminimum expression of 10 counts are dropped.",
  random_mask_cell_cycle_2="Prediction error on dropped values (log scaled)\n10,000 entries of CELL-CYCLE dataset from \ntop %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_cite_cd8_1="Prediction error on dropped values (log scaled)\n1,000 entries of CITE-CD8 dataset with \nminimum expression of 10 UMIs are dropped.",
  random_mask_cite_cd8_2="Prediction error on dropped values (log scaled)\n1,000 entries of CITE-CD8 dataset from \ntop %10 HGV genes with minimum expression of 10 UMIs are dropped.",
  random_mask_cortex_1="Prediction error on dropped values (log scaled)\n10,000 entries of CORTEX dataset with \nminimum expression of 10 UMIs are dropped.",
  random_mask_cortex_2="Prediction error on dropped values (log scaled)\n10,000 entries of CORTEX dataset from \ntop %10 HGV genes with minimum expression of 10 UMIs are dropped.",
  random_mask_pollen_hq_1="Prediction error on dropped values (log scaled)\n10,000 entries of POLLEN-HQ dataset with \nminimum expression of 10 counts are dropped.",
  random_mask_pollen_hq_2="Prediction error on dropped values (log scaled)\n10,000 entries of POLLEN-HQ dataset from \ntop %10 HGV genes with minimum expression of 10 counts are dropped.",
  random_mask_pollen_lq_1="Prediction error on dropped values (log scaled)\n10,000 entries of POLLEN-LQ dataset with \nminimum expression of 10 counts are dropped.",
  random_mask_pollen_lq_2="Prediction error on dropped values (log scaled)\n10,000 entries of POLLEN-LQ dataset from \ntop %10 HGV genes with minimum expression of 10 counts are dropped."
)


dataset.mapping = list(
  random_mask_10xPBMC4k_1="10x-PBMC4k",
  random_mask_10xPBMC4k_2="10x-PBMC4k",
  random_mask_baron_human_1="BARON-HUMAN",
  random_mask_baron_human_2="BARON-HUMAN",
  random_mask_baron_mouse_1="BARON-MOUSE",
  random_mask_baron_mouse_2="BARON-MOUSE",
  random_mask_cell_cycle_1="CELL-CYCLE",
  random_mask_cell_cycle_2="CELL-CYCLE",
  random_mask_cite_cd8_1="CITE-CD8",
  random_mask_cite_cd8_2="CITE-CD8",
  random_mask_cortex_1="CORTEX",
  random_mask_cortex_2="CORTEX",
  random_mask_pollen_hq_1="POLLEN-HQ",
  random_mask_pollen_hq_2="POLLEN-HQ",
  random_mask_pollen_lq_1="POLLEN-LQ",
  random_mask_pollen_lq_2="POLLEN-LQ"
)

dataset.mapping.mapping = list(
  random_mask_10xPBMC4k_1="10x-PBMC4k",
  random_mask_10xPBMC4k_2="10x-PBMC4k (from top %10 HGVs)",
  random_mask_baron_human_1="BARON-HUMAN",
  random_mask_baron_human_2="BARON-HUMAN (from top %10 HGVs)",
  random_mask_baron_mouse_1="BARON-MOUSE",
  random_mask_baron_mouse_2="BARON-MOUSE (from top %10 HGVs)",
  random_mask_cell_cycle_1="CELL-CYCLE",
  random_mask_cell_cycle_2="CELL-CYCLE (from top %10 HGVs)",
  random_mask_cite_cd8_1="CITE-CD8",
  random_mask_cite_cd8_2="CITE-CD8 (from top %10 HGVs)",
  random_mask_cortex_1="CORTEX dataset",
  random_mask_cortex_2="CORTEX dataset (from top %10 HGVs)",
  random_mask_pollen_hq_1="POLLEN-HQ",
  random_mask_pollen_hq_2="POLLEN-HQ (from top %10 HGVs)",
  random_mask_pollen_lq_1="POLLEN-LQ",
  random_mask_pollen_lq_2="POLLEN-LQ (from top %10 HGVs)."
)



dataset.umi.mapping = list(
  random_mask_10xPBMC4k_1=T,
  random_mask_10xPBMC4k_2=T,
  random_mask_baron_human_1=F,
  random_mask_baron_human_2=F,
  random_mask_baron_mouse_1=F,
  random_mask_baron_mouse_2=F,
  random_mask_cell_cycle_1=F,
  random_mask_cell_cycle_2=F,
  random_mask_cite_cd8_1=T,
  random_mask_cite_cd8_2=T,
  random_mask_cortex_1=T,
  random_mask_cortex_2=T,
  random_mask_pollen_hq_1=F,
  random_mask_pollen_hq_2=F,
  random_mask_pollen_lq_1=F,
  random_mask_pollen_lq_2=F
)
dataset.hvg.mapping = list(
  random_mask_10xPBMC4k_1="all",
  random_mask_10xPBMC4k_2="hvg",
  random_mask_baron_human_1="all",
  random_mask_baron_human_2="hvg",
  random_mask_baron_mouse_1="all",
  random_mask_baron_mouse_2="hvg",
  random_mask_cell_cycle_1="all",
  random_mask_cell_cycle_2="hvg",
  random_mask_cite_cd8_1="all",
  random_mask_cite_cd8_2="hvg",
  random_mask_cortex_1="all",
  random_mask_cortex_2="hvg",
  random_mask_pollen_hq_1="all",
  random_mask_pollen_hq_2="hvg",
  random_mask_pollen_lq_1="all",
  random_mask_pollen_lq_2="hvg"
)

dataset.n_drop.mapping = list(
  random_mask_10xPBMC4k_1=10000,
  random_mask_10xPBMC4k_2=10000,
  random_mask_baron_human_1=10000,
  random_mask_baron_human_2=10000,
  random_mask_baron_mouse_1=10000,
  random_mask_baron_mouse_2=10000,
  random_mask_cell_cycle_1=10000,
  random_mask_cell_cycle_2=10000,
  random_mask_cite_cd8_1=1000,
  random_mask_cite_cd8_2=1000,
  random_mask_cortex_1=10000,
  random_mask_cortex_2=10000,
  random_mask_pollen_hq_1=10000,
  random_mask_pollen_hq_2=10000,
  random_mask_pollen_lq_1=10000,
  random_mask_pollen_lq_2=10000
)



RMSE=all.results[all.results$criteria == "RMSE_log",]
RMSE$criteria = NULL
RMSE$n_drop = sapply(RMSE$data_set, function(x) {dataset.n_drop.mapping[[x]]})
RMSE$target = sapply(RMSE$data_set, function(x) {dataset.hvg.mapping[[x]]})
RMSE$data_set = sapply(RMSE$data_set, function(x) {dataset.mapping[[x]]})
RMSE_casted = dcast(RMSE, data_set + target + n_drop ~ method)
RMSE_casted = RMSE_casted[, c(1, 2, 3, order(-100 * (colnames(RMSE_casted[-c(1, 2, 3)]) == "FNN")
  - 10 * colSums(apply(RMSE_casted[-c(1, 2, 3)], 1, FUN=min, na.rm=TRUE) == RMSE_casted[-c(1, 2, 3)]) + 
    colMeans(RMSE_casted[-c(1, 2, 3)], na.rm = T)
) + 3)]
RMSE_casted = RMSE_casted[order(RMSE_casted$target, RMSE_casted$data_set),]
RMSE_casted[-c(1,2,3)] = round(RMSE_casted[-c(1,2,3)], 2)
print(xtable(t(RMSE_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/random_mask/RMSE.tex")

MAE=all.results[all.results$criteria == "MAE_log",]
MAE$criteria = NULL
MAE$n_drop = sapply(MAE$data_set, function(x) {dataset.n_drop.mapping[[x]]})
MAE$target = sapply(MAE$data_set, function(x) {dataset.hvg.mapping[[x]]})
MAE$data_set = sapply(MAE$data_set, function(x) {dataset.mapping[[x]]})
MAE$data_set = paste0("", MAE$data_set,"\n")
MAE_casted = dcast(MAE, data_set + target + n_drop ~ method)
MAE_casted = MAE_casted[, c(1, 2, 3, order(-100 * (colnames(MAE_casted[-c(1, 2, 3)]) == "FNN")
  - 10 * colSums(apply(MAE_casted[-c(1, 2, 3)], 1, FUN=min, na.rm=TRUE) == MAE_casted[-c(1, 2, 3)]) + 
    colMeans(MAE_casted[-c(1, 2, 3)], na.rm = T)
) + 3)]
MAE_casted = MAE_casted[order(MAE_casted$target, MAE_casted$data_set),]
MAE_casted[-c(1,2,3)] = round(MAE_casted[-c(1,2,3)], 2)
print(xtable(t(MAE_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/random_mask/MAE.tex")


preds = all.predictions

for (data.set in unique(all.predictions$data_set)) {
  for (method in unique(all.predictions$method)) {
    if (length(preds[preds$method == method & preds$data_set == data.set, ]$method) == 0) {
      preds[nrow(preds) + 1,] = list(0, 0, method, data.set)
    }
  }
}

# preds$predicted = pmin(preds$predicted, 5 * max(preds$original))
preds$hvg = sapply(preds$data_set, function(x) {dataset.hvg.mapping[[x]]})
preds$method = factor(preds$method, levels=c(colnames(MAE_casted)[-c(1, 2, 3)]))

show.strip = T

sc.scatter.plot = function(data.set) {
  some.predictions = preds[preds$data_set == data.set,]
  some.predictions = some.predictions[some.predictions$method != "no imputation",]
  
  alpha = 0.03
  if (data.set %in% c("random_mask_cite_cd8_1", "random_mask_cite_cd8_2")) {
    alpha = 0.3
  }
  if (show.strip) {
    show.strip.element = element_text(size=17, margin=margin(3, 3, 3, 3, "mm"))
  } else {
    show.strip.element = element_blank()
  }
  predictions.plot = ggplot(some.predictions, aes(x=log(1+original), y=log(1+predicted))) + 
    geom_point(shape=8, size = 1, color="dodgerblue2", alpha=alpha) + 
    expand_limits(x = -0.1, y=-0.1) + geom_abline(slope=1, intercept=0, color="red", alpha=0.2) +
    expand_limits(y = max(log(1 + some.predictions$original))) + 
    facet_wrap(~ method, scales="free_y", nrow=length(unique(preds$method)), strip.position="right",) + theme_bw() +
    theme(panel.grid = element_blank(), aspect.ratio = 1, legend.position = "none", axis.title = element_blank(),
          strip.background=element_blank(), plot.title = element_text(hjust = 0.5, size = 17),
          panel.spacing.y=unit(1, "lines"), strip.text=show.strip.element) +
    ggtitle(dataset.mapping[[data.set]])
  # ylab("Predicted values for zeroed out entries (log scaled)") + 
  # xlab("Original expression values before dropping (log scaled)")
}
  
show.strip = T
for (data.set in unique(preds$data_set)) {
  predictions.plot = sc.scatter.plot(data.set)
  
  pdf(paste0("./output/random_mask/", data.set, ".csv.scatter.plot.pdf"), width = 3, height = 26)
  print(predictions.plot)
  dev.off()
}

hvg_datasets = unique(preds$data_set)[sapply(unique(preds$data_set), function(x) {dataset.hvg.mapping[[x]]}) == "hvg"]
complete_datasets = unique(preds$data_set)[sapply(unique(preds$data_set), function(x) {dataset.hvg.mapping[[x]]}) == "all"]


show.strip = F
pl <- lapply(hvg_datasets, sc.scatter.plot)
show.strip = T
pl[[length(pl)]] = sc.scatter.plot(hvg_datasets[length(hvg_datasets)])

pdf(paste0("./output/random_mask/all.hvg.scatter.plot.pdf"), width = 19.75, height = 26)
grid.arrange(grobs=pl, nrow=1, ncol=length(hvg_datasets), widths = c(rep(2.5, 7), 3))
dev.off()



show.strip = F
pl <- lapply(complete_datasets, sc.scatter.plot)
show.strip = T
pl[[length(pl)]] = sc.scatter.plot(complete_datasets[length(complete_datasets)])

pdf(paste0("./output/random_mask/all.all.scatter.plot.pdf"), width = 19.75, height = 26)
grid.arrange(grobs=pl, nrow=1, ncol=length(complete_datasets), widths = c(rep(2.5, 7), 3))
dev.off()



sc.plot.density = function(data.set) {
  some.predictions = preds[preds$data_set == data.set,]
  some.predictions = some.predictions[some.predictions$method != "no imputation",]
  some.predictions$method = factor(some.predictions$method, levels=c(colnames(MAE_casted)[-c(1, 2, 3)]))
  some.predictions$original = log(1+some.predictions$original)
  some.predictions$predicted = log(1+some.predictions$predicted)
  some.predictions$predicted[some.predictions$predicted < 3e-2] = runif(sum(some.predictions$predicted < 3e-2), -3e-2, 3e-2)
  some.predictions$original[some.predictions$predicted < 3e-2] = some.predictions$original[some.predictions$predicted < 3e-2] + runif(sum(some.predictions$predicted < 3e-2), -3e-2, 3e-2)
  
  if (show.strip) {
    show.strip.element = element_text(size=17, margin=margin(3, 3, 3, 3, "mm"))
  } else {
    show.strip.element = element_blank()
  }
  
  predictions.plot = ggplot(some.predictions, 
                            aes(x=original, y=predicted)) +
    stat_density_2d(aes(fill=stat(ndensity)), geom = "raster", contour = FALSE) +
    expand_limits(x = -0.1, y=-0.1) + geom_abline(slope=1, intercept=0, color="red", alpha=0.2) +
    facet_wrap(~ method, scales="free_y", nrow=length(unique(preds$method)), strip.position="right",) + theme_bw() +
    expand_limits(y = max(some.predictions$original)) + 
    scale_fill_material("blue") +
    # scale_fill_distiller(palette= "Blues", direction=1) + 
    # scale_colour_gradient(low = "white", high = "dodgerblue2") +
    theme(panel.grid = element_blank(), aspect.ratio = 1, legend.position = "none", axis.title = element_blank(),
          strip.background=element_blank(), plot.title = element_text(hjust = 0.5, size = 17),
          panel.spacing.y=unit(1, "lines"), strip.text=show.strip.element) +
    ggtitle(dataset.mapping[[data.set]])
}

show.strip = T
for (data.set in unique(preds$data_set)) {
  predictions.plot = sc.plot.density(data.set)
  
  pdf(paste0("./output/random_mask/", data.set, ".csv.density.plot.pdf"), width = 3, height = 26)
  print(predictions.plot)
  dev.off()
}

show.strip = F
pl <- lapply(hvg_datasets, sc.plot.density)
show.strip = T
pl[[length(pl)]] = sc.plot.density(hvg_datasets[length(hvg_datasets)])

pdf(paste0("./output/random_mask/all.hvg.density.plot.pdf"), width = 19.75, height = 26)
grid.arrange(grobs=pl, nrow=1, ncol=length(hvg_datasets), widths = c(rep(2.5, 7), 3))
dev.off()



show.strip = F
pl <- lapply(complete_datasets, sc.plot.density)
show.strip = T
pl[[length(pl)]] = sc.plot.density(complete_datasets[length(complete_datasets)])

pdf(paste0("./output/random_mask/all.all.density.plot.pdf"), width = 19.75, height = 26)
grid.arrange(grobs=pl, nrow=1, ncol=length(complete_datasets), widths = c(rep(2.5, 7), 3))
dev.off()

# 
# 
# 
# some.predictions = all.predictions[all.predictions$method != "no imputation",]
# # some.predictions = some.predictions[some.predictions$method %in% c("DCA", "ZINB-WaVE", "scVI", "SAVER"),]
# # some.predictions = some.predictions[some.predictions$data_set %in% c("random_mask_pollen_hq_2", "random_mask_cell_cycle_1", "random_mask_10xPBMC4k_2", "random_mask_baron_mouse_1"),]
# some.predictions$original = log(1+some.predictions$original)
# some.predictions$predicted = log(1+some.predictions$predicted)
# some.predictions$predicted[some.predictions$predicted < 1e-1] = runif(sum(some.predictions$predicted < 1e-1), -1e-1, 1e-1)
# some.predictions$original[some.predictions$predicted < 1e-1] = some.predictions$original[some.predictions$predicted < 1e-1] + runif(sum(some.predictions$predicted < 1e-1), -1e-1, 1e-1)
# some.predictions$predicted = pmin(some.predictions$predicted, 3 * max(some.predictions$original))
# some.predictions$method = factor(some.predictions$method, levels=c(colnames(MAE_casted)[-c(1, 2, 3)]))
# some.predictions$hvg = sapply(some.predictions$data_set, function(x) {dataset.hvg.mapping[[x]]})
# some.predictions$data_set = sapply(some.predictions$data_set, function(x) {dataset.mapping[[x]]})
# 
# 
# pdf(paste0("./output/random_mask/predictions.density.all.plot.pdf"), width = 21, height = 30)
# predictions.plot = ggplot(some.predictions[some.predictions$hvg == "all", ], 
#                           aes(x=original, y=predicted)) +
#   stat_density_2d(aes(fill=stat(ndensity)), geom = "raster", contour = FALSE) +
#   expand_limits(x = -0.1, y=-0.1) + geom_abline(slope=1, intercept=0, color="red", alpha=0.2) +
#   facet_grid(rows = vars(method), cols = vars(data_set), scale="free") + theme_bw() +
#   scale_fill_material("blue") +
#   # scale_fill_distiller(palette= "Blues", direction=1) + 
#   # scale_colour_gradient(low = "white", high = "dodgerblue2") + 
#   theme(panel.grid = element_blank(), aspect.ratio = 1,
#         strip.background=element_blank(),
#         strip.text=element_text(size=9, margin=margin(1, 1, 1, 1, "mm")))
# print(predictions.plot)
# dev.off()
# 
# pdf(paste0("./output/random_mask/predictions.density.hvg.plot.pdf"), width = 21, height = 30)
# predictions.plot = ggplot(some.predictions[some.predictions$hvg == "hvg", ], 
#                           aes(x=original, y=predicted)) +
#   stat_density_2d(aes(fill=stat(ndensity)), geom = "raster", contour = FALSE) +
#   expand_limits(x = -0.1, y=-0.1) + geom_abline(slope=1, intercept=0, color="red", alpha=0.2) +
#   facet_grid(rows = vars(method), cols = vars(data_set), scale="free") + theme_bw() +
#   scale_fill_material("blue") +
#   # scale_fill_distiller(palette= "Blues", direction=1) + 
#   # scale_colour_gradient(low = "white", high = "dodgerblue2") + 
#   theme(panel.grid = element_blank(), aspect.ratio = 1,
#         strip.background=element_blank(),
#         strip.text=element_text(size=9, margin=margin(1, 1, 1, 1, "mm")))
# print(predictions.plot)
# dev.off()


for (data.set in unique(all.results$data_set)) {
    some.results = all.results[all.results$data_set == data.set,]
    some.results = some.results[some.results$criteria %in% c("RMSE_log", "MAE_log"),]
    some.results$criteria[some.results$criteria == "RMSE_log"] = "RMSE"
    some.results$criteria[some.results$criteria == "MAE_log"] = "MAE"
    results.plot = ggplot(some.results, aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=criteria)) + 
      geom_bar(stat="identity", position=position_dodge()) +
      theme_bw() + scale_fill_brewer(palette="Set1") +
      theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
      ggtitle(title.mapping.2[[data.set]]) + theme(plot.title = element_text(size = 12)) +
      ylab("") + 
      xlab("")
    
    pdf(paste0("./output/random_mask/", data.set, ".csv.error.plot.pdf"), width = 8, height = 5)
    print(results.plot)
    dev.off()
}


some.results = all.results
some.results = some.results[some.results$criteria %in% c("RMSE_log", "MAE_log"),]
some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping.mapping[[x]]})
some.results$criteria[some.results$criteria == "RMSE_log"] = "RMSE"
some.results$criteria[some.results$criteria == "MAE_log"] = "MAE"
results.plot = ggplot(some.results, aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=criteria)) + 
  geom_bar(stat="identity", position=position_dodge()) +
  facet_wrap(~ data_set, scales="free_y", nrow=3) + theme_bw() + scale_fill_brewer(palette="Set1") +
  theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
  ggtitle("Prediction error on dropped values") + theme(plot.title = element_text(size = 15)) +
  ylab("") + 
  xlab("")

pdf(paste0("./output/random_mask/all.data.sets.error.plot.pdf"), width = 13, height = 9)
print(results.plot)
dev.off()


plot.errors = function(criteria, hvg, ylab) {
  some.results = all.results
  some.results = some.results[some.results$criteria == criteria,]
  some.results$hvg = sapply(some.results$data_set, function(x) {dataset.hvg.mapping[[x]]})
  some.results$method = factor(some.results$method, levels=c(colnames(MAE_casted)[-c(1, 2, 3)]))
  some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping[[x]]})
  some.results = some.results[some.results$hvg == hvg, ]
  
  results.plot = ggplot(some.results,
                        aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=data_set)) + 
    geom_bar(stat="identity", position=position_dodge(), width=0.7) +
    theme_bw() +  scale_color_igv () + facet_wrap(~data_set, ncol = 4, scales = "free") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1, size=8)) +
    # ggtitle("Prediction error on dropped values") + theme(plot.title = element_text(size = 15)) +
    theme(legend.position = "none", strip.background=element_blank(), 
          plot.title = element_text(hjust = 0.5, size = 9),
          panel.spacing=unit(1, "lines"), strip.text=element_text(size=9, margin=margin(3, 3, 3, 3, "mm")),
          axis.title.y = element_text(size=9, hjust = 0.9, angle = 90)) +
    labs(fill = "Data-Set") + scale_fill_d3(palette = "category10") +
    ylab(ylab) + 
    xlab("")
  
  pdf(paste0("./output/random_mask/", criteria, ".", hvg, ".pdf"), width = 10, height = 7)
  print(results.plot)
  dev.off()
}


plot.errors("MAE_log", "all", "Mean Absolute Error")
plot.errors("MAE_log", "hvg", "Mean Absolute Error")
plot.errors("RMSE_log", "all", "Root Mean Squared Error")
plot.errors("RMSE_log", "hvg", "Root Mean Squared Error")



plot.normal.errors = function(criteria, hvg, ylab) {
  
  some.results = all.results
  some.results = some.results[some.results$criteria == criteria,]
  some.results$hvg = sapply(some.results$data_set, function(x) {dataset.hvg.mapping[[x]]})
  some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping[[x]]})
  some.results = some.results[some.results$hvg == hvg, ]
  some.results = dcast(some.results, data_set + hvg ~ method)
  some.results[-c(1,2)] = some.results[-c(1,2)] / some.results[["no imputation"]]
  some.results = melt(some.results, id.vars = c("data_set", "hvg"), variable.name="method")
  some.results$method = factor(some.results$method, levels=c(colnames(MAE_casted)[-c(1, 2, 3)]))
  
  results.plot = ggplot(some.results,
                        aes(x=reorder(reorder(method, value), method!="FNN"), y=value)) + 
    geom_violin(width=0.7, fill="gray") +
    geom_point(aes(group=data_set, color=data_set), size=1) +
    # geom_line(aes(group=data_set, color=data_set)) +
    theme_bw() +  scale_color_igv () +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1, size=8)) +
    # ggtitle("Prediction error on dropped values") + theme(plot.title = element_text(size = 15)) +
    theme(strip.background=element_blank(), 
          plot.title = element_text(hjust = 0.5, size = 9),
          panel.spacing=unit(1, "lines"), strip.text=element_text(size=9, margin=margin(3, 3, 3, 3, "mm")),
          axis.title.y = element_text(size=9, angle = 90)) +
    labs(fill = "Data-Set") + scale_fill_d3(palette = "category10") +
    ylab(ylab) + 
    xlab("")
  
  pdf(paste0("./output/random_mask/", criteria, ".", hvg, ".normalized.violin.pdf"), width = 8, height = 4.5)
  print(results.plot)
  dev.off()
}


plot.normal.errors("MAE_log", "all", "Normalized Mean Absolute Error")
plot.normal.errors("MAE_log", "hvg", "Normalized Mean Absolute Error")
plot.normal.errors("RMSE_log", "all", "Normalized Root Mean Squared Error")
plot.normal.errors("RMSE_log", "hvg", "Normalized Root Mean Squared Error")




plot.viol.errors = function(criteria, hvg, ylab) {
  
  some.results = all.results
  some.results = some.results[some.results$criteria == criteria,]
  some.results$hvg = sapply(some.results$data_set, function(x) {dataset.hvg.mapping[[x]]})
  some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping[[x]]})
  some.results = some.results[some.results$hvg == hvg, ]
  some.results$method = factor(some.results$method, levels=c(colnames(MAE_casted)[-c(1, 2, 3)]))
  
  results.plot = ggplot(some.results,
                        aes(x=reorder(reorder(method, value), method!="FNN"), y=value)) + 
    geom_violin(width=0.7, fill="gray") +
    geom_point(aes(group=data_set, color=data_set), size=1) +
    # geom_line(aes(group=data_set, color=data_set)) +
    theme_bw() +  scale_color_igv () +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1, size=8)) +
    # ggtitle("Prediction error on dropped values") + theme(plot.title = element_text(size = 15)) +
    theme(strip.background=element_blank(), 
          plot.title = element_text(hjust = 0.5, size = 9),
          panel.spacing=unit(1, "lines"), strip.text=element_text(size=9, margin=margin(3, 3, 3, 3, "mm")),
          axis.title.y = element_text(size=9, angle = 90)) +
    labs(fill = "Data-Set") + scale_fill_d3(palette = "category10") +
    ylab(ylab) + 
    xlab("")
  
  pdf(paste0("./output/random_mask/", criteria, ".", hvg, ".violin.pdf"), width = 8, height = 4.5)
  print(results.plot)
  dev.off()
}


plot.viol.errors("MAE_log", "all", "Mean Absolute Error")
plot.viol.errors("MAE_log", "hvg", "Mean Absolute Error")
plot.viol.errors("RMSE_log", "all", "Root Mean Squared Error")
plot.viol.errors("RMSE_log", "hvg", "Root Mean Squared Error")



