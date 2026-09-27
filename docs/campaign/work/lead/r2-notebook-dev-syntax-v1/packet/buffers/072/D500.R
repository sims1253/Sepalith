".fac" <- function(x) {

  # exact factorial; NB returns (by design) a
  # bigz; used in setparts()
  x <- as.integer(x)
  if (x < 0) {
    stop("x must be non-negative")
  }
  if (x == 0) {
    return(1)
  }
  fac <- 1
  for (i in 1:x) {
    fac <- fac * i
  }
  return(fac)
}