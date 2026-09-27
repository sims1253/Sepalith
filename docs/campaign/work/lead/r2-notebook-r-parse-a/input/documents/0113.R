vech <- function(x) {
  x <- as.matrix(x)

  if (dim(x)[1] != dim(x)[2]) {
    stop("Non-square matrix passed to vech().\n")
  }

  output <- x[lower.tri(x, diag = TRUE)]
  dim(output) <- NULL
  return(output)
}