est.matrix <- function(x, name) {
  vals <- lapply(x, function(z) z[[name]])
  out <- do.call(cbind, vals)
  out
}