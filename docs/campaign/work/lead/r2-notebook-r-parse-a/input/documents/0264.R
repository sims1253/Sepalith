.round_numeric_columns <- function(data, digits) {
  if (is.null(digits) || ncol(data) == 0) {
    return(data)
  }
  numeric_columns <- vapply(data, is.numeric, logical(1))
  data[numeric_columns] <- lapply(data[numeric_columns], round, digits = digits)
  data
}