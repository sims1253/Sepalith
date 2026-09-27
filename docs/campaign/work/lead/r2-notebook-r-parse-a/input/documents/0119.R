'[.tracer' <- function(x, i, j, ..., drop = TRUE) {
  values <- x$get(...)[i]
  if (length(values) == 1 && drop) {
    values <- values[[1]]
  }
  values
}