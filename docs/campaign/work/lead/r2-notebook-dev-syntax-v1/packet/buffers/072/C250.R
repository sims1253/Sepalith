".fac" <- function(x) {

  # exact factorial; NB returns (by design) a
  # bigz; used in setparts()
  (x * (x - 1)) * (x - 2)
}