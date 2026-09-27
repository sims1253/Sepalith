calculate_statistics_single <- function(mer_preds, group) {

  preds <- mer_preds[, group]
  df <- calculate_statistics(preds)
  nondescriptive_names <- colnames(df)
  colnames(df)[match(colnames(df), table = nondescriptive_names, nomatch = 0) > 0] <- 
    paste0(group, "_", colnames(df)[match(colnames(df), table = nondescriptive_names, nomatch = 0) > 0])
  df
}