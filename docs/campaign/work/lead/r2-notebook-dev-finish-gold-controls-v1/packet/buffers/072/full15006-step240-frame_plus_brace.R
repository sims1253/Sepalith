".fac" <- function(x) {

  # exact factorial; NB returns (by design) a
  # bigz; used in setparts()
  if (length(x) == 1) {
    return(x)
  }
  if (length(x) == 2) {
    return(x[1] * x[2])
  }
  return(prod(x))

}