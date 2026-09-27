".fac" <- function(x) {

  # exact factorial; NB returns (by design) a
  # bigz; used in setparts()
  if (is.numeric(x)) {
    x <- as.factor(x)
  }
  if (is.factor(x)) {
    x <- as.numeric(x)
  }
  if (is.numeric(x)) {
    x <- as.factor(x)
  }
  x
}