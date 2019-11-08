
library(stringr)
library(ggplot2)
library(reshape2)
library(xtable)
library(ggsci)




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
  down_sample_10xPBMC4k_1="10x-PBMC4k (%10 preserved)",
  down_sample_10xPBMC4k_2="10x-PBMC4k (%50 preserved)",
  down_sample_baron_human_1="BARON-HUMAN (%10 preserved)",
  down_sample_baron_human_2="BARON-HUMAN (%50 preserved)",
  down_sample_baron_mouse_1="BARON-MOUSE (%10 preserved)",
  down_sample_baron_mouse_2="BARON-MOUSE (%50 preserved)",
  down_sample_cell_cycle_1="CELL-CYCLE (%10 preserved)",
  down_sample_cell_cycle_2="CELL-CYCLE (%50 preserved)",
  down_sample_cite_cd8_1="CITE-CD8 (%10 preserved)",
  down_sample_cite_cd8_2="CITE-CD8 (%50 preserved)",
  down_sample_cortex_1="CORTEX (%10 preserved)",
  down_sample_cortex_2="CORTEX (%50 preserved)",
  down_sample_pollen_hq_1="POLLEN-HQ (%10 preserved)",
  down_sample_pollen_hq_2="POLLEN-HQ (%50 preserved)",
  down_sample_pollen_lq_1="POLLEN-HQ (%10 preserved)",
  down_sample_pollen_lq_2="POLLEN-HQ (%50 preserved)."
)

dataset.mapping = list(
  down_sample_10xPBMC4k_1="10x-PBMC4k",
  down_sample_10xPBMC4k_2="10x-PBMC4k",
  down_sample_baron_human_1="BARON-HUMAN",
  down_sample_baron_human_2="BARON-HUMAN",
  down_sample_baron_mouse_1="BARON-MOUSE",
  down_sample_baron_mouse_2="BARON-MOUSE",
  down_sample_cell_cycle_1="CELL-CYCLE",
  down_sample_cell_cycle_2="CELL-CYCLE",
  down_sample_cite_cd8_1="CITE-CD8",
  down_sample_cite_cd8_2="CITE-CD8",
  down_sample_cortex_1="CORTEX",
  down_sample_cortex_2="CORTEX",
  down_sample_pollen_hq_1="POLLEN-HQ",
  down_sample_pollen_hq_2="POLLEN-HQ",
  down_sample_pollen_lq_1="POLLEN-LQ",
  down_sample_pollen_lq_2="POLLEN-LQ"
)

dataset.ratio.mapping = list(
  down_sample_10xPBMC4k_1="10",
  down_sample_10xPBMC4k_2="50",
  down_sample_baron_human_1="10",
  down_sample_baron_human_2="50",
  down_sample_baron_mouse_1="10",
  down_sample_baron_mouse_2="50",
  down_sample_cell_cycle_1="10",
  down_sample_cell_cycle_2="50",
  down_sample_cite_cd8_1="10",
  down_sample_cite_cd8_2="50",
  down_sample_cortex_1="10",
  down_sample_cortex_2="50",
  down_sample_pollen_hq_1="10",
  down_sample_pollen_hq_2="50",
  down_sample_pollen_lq_1="10",
  down_sample_pollen_lq_2="50"
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

if (!file.exists("./down_sample.rds")) {
  
  
  test.results.dir = "/home/amirali/code/test\\ results"
  results = system(paste0("ls ", test.results.dir, "/*/down_sample_*/result.txt"), intern = TRUE)
  results = c(results, system(paste0("ls ", "/home/amirali/code/cavatappi/results", "/*/down_sample_*/result.txt"), intern = TRUE))
  
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
  
  
  saveRDS(list(all.results), file = "./down_sample.rds")
} else {
  data = readRDS(file = "./down_sample.rds")
  all.results = data[[1]]
}



all.results = all.results[all.results$method !=  "NA", ]


title.mapping = list(
  down_sample_10xPBMC4k_1="Recovery of original values given subsampled data (log scaled)\n10% of UMIs from 10x-PBMC4k dataset are preserve.",
  down_sample_10xPBMC4k_2="Recovery of original values given subsampled data (log scaled)\n50% of UMIs from 10x-PBMC4k dataset are preserved.",
  down_sample_baron_human_1="Recovery of original values given subsampled data (log scaled)\n10% of counts from BARON-HUMAN dataset are preserved.",
  down_sample_baron_human_2="Recovery of original values given subsampled data (log scaled)\n50% of counts from BARON-HUMAN dataset are preserved.",
  down_sample_baron_mouse_1="Recovery of original values given subsampled data (log scaled)\n10% of counts from BARON-MOUSE dataset are preserved.",
  down_sample_baron_mouse_2="Recovery of original values given subsampled data (log scaled)\n50% of counts from BARON-MOUSE dataset are preserved.",
  down_sample_cell_cycle_1="Recovery of original values given subsampled data (log scaled)\n10% of counts from CELL-CYCLE dataset are preserved.",
  down_sample_cell_cycle_2="Recovery of original values given subsampled data (log scaled)\n50% of counts from CELL-CYCLE dataset are preserved.",
  down_sample_cite_cd8_1="Recovery of original values given subsampled data (log scaled)\n10% of UMIs from CITE-CD8 dataset are preserved.",
  down_sample_cite_cd8_2="Recovery of original values given subsampled data (log scaled)\n50% of UMIs from CITE-CD8 dataset are preserved.",
  down_sample_cortex_1="Recovery of original values given subsampled data (log scaled)\n10% of UMIs from CORTEX dataset dataset are preserved.",
  down_sample_cortex_2="Recovery of original values given subsampled data (log scaled)\n50% of UMIs from CORTEX dataset dataset are preserved.",
  down_sample_pollen_lq_1="Recovery of original values given subsampled data (log scaled)\n10% of counts from POLLEN-LQ dataset are preserved.",
  down_sample_pollen_lq_2="Recovery of original values given subsampled data (log scaled)\n50% of counts from POLLEN-LQ dataset are preserved."
)


dataset.mapping = list(
  down_sample_10xPBMC4k_1="10x-PBMC4k (%10 preserved)",
  down_sample_10xPBMC4k_2="10x-PBMC4k (%50 preserved)",
  down_sample_baron_human_1="BARON-HUMAN (%10 preserved)",
  down_sample_baron_human_2="BARON-HUMAN (%50 preserved)",
  down_sample_baron_mouse_1="BARON-MOUSE (%10 preserved)",
  down_sample_baron_mouse_2="BARON-MOUSE (%50 preserved)",
  down_sample_cell_cycle_1="CELL-CYCLE (%10 preserved)",
  down_sample_cell_cycle_2="CELL-CYCLE (%50 preserved)",
  down_sample_cite_cd8_1="CITE-CD8 (%10 preserved)",
  down_sample_cite_cd8_2="CITE-CD8 (%50 preserved)",
  down_sample_cortex_1="CORTEX (%10 preserved)",
  down_sample_cortex_2="CORTEX (%50 preserved)",
  down_sample_pollen_hq_1="POLLEN-HQ (%10 preserved)",
  down_sample_pollen_hq_2="POLLEN-HQ (%50 preserved)",
  down_sample_pollen_lq_1="POLLEN-HQ (%10 preserved)",
  down_sample_pollen_lq_2="POLLEN-HQ (%50 preserved)."
)

dataset.mapping = list(
  down_sample_10xPBMC4k_1="10x-PBMC4k",
  down_sample_10xPBMC4k_2="10x-PBMC4k",
  down_sample_baron_human_1="BARON-HUMAN",
  down_sample_baron_human_2="BARON-HUMAN",
  down_sample_baron_mouse_1="BARON-MOUSE",
  down_sample_baron_mouse_2="BARON-MOUSE",
  down_sample_cell_cycle_1="CELL-CYCLE",
  down_sample_cell_cycle_2="CELL-CYCLE",
  down_sample_cite_cd8_1="CITE-CD8",
  down_sample_cite_cd8_2="CITE-CD8",
  down_sample_cortex_1="CORTEX",
  down_sample_cortex_2="CORTEX",
  down_sample_pollen_hq_1="POLLEN-HQ",
  down_sample_pollen_hq_2="POLLEN-HQ",
  down_sample_pollen_lq_1="POLLEN-LQ",
  down_sample_pollen_lq_2="POLLEN-LQ"
)

dataset.ratio.mapping = list(
  down_sample_10xPBMC4k_1="10",
  down_sample_10xPBMC4k_2="50",
  down_sample_baron_human_1="10",
  down_sample_baron_human_2="50",
  down_sample_baron_mouse_1="10",
  down_sample_baron_mouse_2="50",
  down_sample_cell_cycle_1="10",
  down_sample_cell_cycle_2="50",
  down_sample_cite_cd8_1="10",
  down_sample_cite_cd8_2="50",
  down_sample_cortex_1="10",
  down_sample_cortex_2="50",
  down_sample_pollen_hq_1="10",
  down_sample_pollen_hq_2="50",
  down_sample_pollen_lq_1="10",
  down_sample_pollen_lq_2="50"
)



RMSE=all.results[all.results$criteria == "all_root_mean_squared_error_on_non_zeros",]
RMSE$criteria = NULL
RMSE$ratio = sapply(RMSE$data_set, function(x) {dataset.ratio.mapping[[x]]})
RMSE = RMSE[RMSE$ratio == "10",]
RMSE$data_set = sapply(RMSE$data_set, function(x) {dataset.mapping[[x]]})
RMSE$data_set = paste0("", RMSE$data_set,"\n")
RMSE_casted = dcast(RMSE, data_set + ratio ~ method)
RMSE_casted = RMSE_casted[, c(1, 2, order(-100 * (colnames(RMSE_casted[-c(1, 2)]) == "FNN")
  - 10 * colSums(apply(RMSE_casted[-c(1, 2)], 1, FUN=min, na.rm=TRUE) == RMSE_casted[-c(1, 2)], na.rm = T) + 
    sapply(RMSE_casted[-c(1, 2)], FUN=median, na.rm=TRUE)
) + 2)]
RMSE_casted = RMSE_casted[order(RMSE_casted$ratio, RMSE_casted$data_set),]
RMSE_casted[-c(1,2)] = round(RMSE_casted[-c(1,2)], 2)
print(xtable(t(RMSE_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/down_sample/RMSE.tex")



MAE=all.results[all.results$criteria == "all_mean_absolute_error_on_non_zeros",]
MAE$criteria = NULL
MAE$ratio = sapply(MAE$data_set, function(x) {dataset.ratio.mapping[[x]]})
MAE = MAE[MAE$ratio == "10",]
MAE$data_set = sapply(MAE$data_set, function(x) {dataset.mapping[[x]]})
MAE$data_set = paste0("", MAE$data_set,"\n")
MAE_casted = dcast(MAE, data_set + ratio ~ method)
MAE_casted = MAE_casted[, c(1, 2, order(-100 * (colnames(RMSE_casted[-c(1, 2)]) == "FNN")
  - 10 * colSums(apply(MAE_casted[-c(1, 2)], 1, FUN=min, na.rm=TRUE) == MAE_casted[-c(1, 2)], na.rm = T) + 
    sapply(MAE_casted[-c(1, 2)], FUN=median, na.rm=TRUE)
) + 2)]
MAE_casted = MAE_casted[order(MAE_casted$ratio, MAE_casted$data_set),]
MAE_casted[-c(1,2)] = round(MAE_casted[-c(1,2)], 2)
print(xtable(t(MAE_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/down_sample/MAE.tex")


MCD=all.results[all.results$criteria == "cell_mean_cosine_distance",]
MCD$criteria = NULL
MCD$ratio = sapply(MCD$data_set, function(x) {dataset.ratio.mapping[[x]]})
MCD = MCD[MCD$ratio == "10",]
MCD$data_set = sapply(MCD$data_set, function(x) {dataset.mapping[[x]]})
MCD$data_set = paste0("", MCD$data_set,"\n")
MCD_casted = dcast(MCD, data_set + ratio ~ method)
MCD_casted = MCD_casted[, c(1, 2, order(-100 * (colnames(RMSE_casted[-c(1, 2)]) == "FNN")
  - 10 * colSums(apply(MCD_casted[-c(1, 2)], 1, FUN=min, na.rm=TRUE) == MCD_casted[-c(1, 2)], na.rm = T) + 
    sapply(MCD_casted[-c(1, 2)], FUN=median, na.rm=TRUE)
) + 2)]
MCD_casted = MCD_casted[order(MCD_casted$ratio, MCD_casted$data_set),]
MCD_casted[-c(1,2)] = round(MCD_casted[-c(1,2)], 2)
print(xtable(t(MCD_casted), sanitize.text.function=identity, include.colnames = F, type = "latex"), file="output/down_sample/MCD.tex")



for (data.set in unique(all.results$data_set)) {
  some.results = all.results[all.results$data_set == data.set,]
  some.results = some.results[some.results$criteria %in% 
                                c("all_mean_absolute_error_on_non_zeros", 
                                  "all_root_mean_squared_error_on_non_zeros"),]
  some.results$criteria[some.results$criteria == "all_root_mean_squared_error_on_non_zeros"] = "RMSE"
  some.results$criteria[some.results$criteria == "all_mean_absolute_error_on_non_zeros"] = "MAE"
  results.plot = ggplot(some.results, aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=criteria)) + 
    geom_bar(stat="identity", position=position_dodge()) +
    theme_bw() + scale_fill_brewer(palette="Set1") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
    ggtitle(title.mapping[[data.set]]) + theme(plot.title = element_text(size = 12)) +
    ylab("") + 
    xlab("")
  
  pdf(paste0("./output/down_sample/", data.set, ".csv.error.plot.pdf"), width = 8, height = 5)
  print(results.plot)
  dev.off()
}

for (data.set in unique(all.results$data_set)) {
  some.results = all.results[all.results$data_set == data.set,]
  some.results = some.results[some.results$criteria %in% 
                                c("cell_mean_cosine_distance"),]
  some.results$criteria[some.results$criteria == "cell_mean_cosine_distance"] = "Mean Cosine Distance"
  results.plot = ggplot(some.results, aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=criteria)) + 
    geom_bar(stat="identity", position=position_dodge()) +
    theme_bw() + scale_fill_brewer(palette="Set1") +
    theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
    ggtitle(title.mapping[[data.set]]) + theme(plot.title = element_text(size = 12)) +
    ylab("") + 
    xlab("")
  
  pdf(paste0("./output/down_sample/", data.set, ".csv.MCD.plot.pdf"), width = 8, height = 5)
  print(results.plot)
  dev.off()
}

some.results = all.results
some.results = some.results[some.results$criteria %in% 
                              c("all_mean_absolute_error_on_non_zeros", 
                                "all_root_mean_squared_error_on_non_zeros"),]
some.results$criteria[some.results$criteria == "all_root_mean_squared_error_on_non_zeros"] = "RMSE"
some.results$criteria[some.results$criteria == "all_mean_absolute_error_on_non_zeros"] = "MAE"
some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping[[x]]})
results.plot = ggplot(some.results, aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=criteria)) + 
  geom_bar(stat="identity", position=position_dodge()) +
  facet_wrap(~ data_set, scales="free_y", nrow=3) + theme_bw() + scale_fill_brewer(palette="Set1") +
  theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
  ggtitle("Prediction error on dropped values (lower is better)") + theme(plot.title = element_text(size = 15)) +
  ylab("") + 
  xlab("")

pdf(paste0("./output/down_sample/all.data.sets.error.plot.pdf"), width = 13, height = 9)
print(results.plot)
dev.off()


some.results = all.results
some.results = some.results[some.results$criteria %in% 
                              c("cell_mean_cosine_distance"),]
some.results$criteria[some.results$criteria == "cell_mean_cosine_distance"] = "Mean Cosine Distance"
some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping[[x]]})
results.plot = ggplot(some.results, aes(x=reorder(reorder(method, value), method!="FNN"), y=value, fill=criteria)) + 
  geom_bar(stat="identity", position=position_dodge()) +
  facet_wrap(~ data_set, scales="free_y", nrow=3) + theme_bw() + scale_fill_brewer(palette="Set1") +
  theme(axis.text.x = element_text(color="black", angle = 45, hjust = 1)) +
  ggtitle("Prediction error on dropped values (lower is better)") + theme(plot.title = element_text(size = 15)) +
  ylab("") + 
  xlab("")

pdf(paste0("./output/down_sample/all.data.sets.MCD.plot.pdf"), width = 13, height = 9)
print(results.plot)
dev.off()






plot.errors = function(criteria, ratio, ylab) {
  some.results = all.results
  some.results = some.results[some.results$criteria == criteria,]
  some.results$ratio = sapply(some.results$data_set, function(x) {dataset.ratio.mapping[[x]]})
  some.results$method = factor(some.results$method, levels=c(colnames(MAE_casted)[-c(1, 2)]))
  some.results$data_set = sapply(some.results$data_set, function(x) {dataset.mapping[[x]]})
  some.results = some.results[some.results$ratio == ratio, ]
  print(some.results)
  
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
  
  pdf(paste0("./output/down_sample/", criteria, ".", ratio, ".pdf"), width = 10, height = 7)
  print(results.plot)
  dev.off()
}


plot.errors("all_mean_absolute_error_on_non_zeros", "10", "Mean Absolute Error")
plot.errors("all_mean_absolute_error_on_non_zeros", "50", "Mean Absolute Error")
plot.errors("all_root_mean_squared_error_on_non_zeros", "10", "Root Mean Squared Error")
plot.errors("all_root_mean_squared_error_on_non_zeros", "50", "Root Mean Squared Error")
plot.errors("cell_mean_cosine_distance", "10", "Root Mean Squared Error")
plot.errors("cell_mean_cosine_distance", "50", "Root Mean Squared Error")







