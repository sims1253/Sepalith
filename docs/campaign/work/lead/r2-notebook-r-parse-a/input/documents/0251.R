gather <- function(data, idvar, col_name, value_name = 'value') {
  not_idvar <- colnames(data)
  not_idvar <- not_idvar[not_idvar != idvar]
  result <- stack(data, select = not_idvar)
  result$group <- as.factor(rep(data[[idvar]], times = ncol(data) - 1))
  colnames(result) <- c(value_name, col_name, idvar)
  result
}