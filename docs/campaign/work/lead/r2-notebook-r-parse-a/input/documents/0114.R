xpnd <- function(x, nrow = NULL) {
  dim(x) <- NULL
  if (is.null(nrow)) {
    nrow <- (-1 + sqrt(1 + 8 * length(x))) / 2
  }
  output <- matrix(0, nrow, nrow)
  output[lower.tri(output, diag = TRUE)] <- x
  hold <- output
  hold[upper.tri(hold, diag = TRUE)] <- 0
  output <- output + t(hold)
  return(output)
}