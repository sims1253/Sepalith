pmf.out <- function(x, p) {
  out <- rep(0, length(x))
  for (i in seq_along(x)) {
    if (as.integer(x[i]) != x[i]) {
      warning(paste("non-integer x = ", x[i]))
      out[i] <- 0
    } else if (x[i] < 0) {
      out[i] <- 0
    } else if (x[i] >= 0) {
      out[i] <- p[x[i] + 1]
    }
  }
  return(out)
}