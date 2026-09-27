calculate_statistics_mc <- function(mer_preds, groups) {
  res <- do.call(cbind,
                 lapply(groups, function(i) {
    calculate_statistics_single(mer_preds, i)
  }))
  stat <- subset(res[,!duplicated(colnames(res))], select = -c(amp_n_peptide, neg_n_peptide))
  stat
}