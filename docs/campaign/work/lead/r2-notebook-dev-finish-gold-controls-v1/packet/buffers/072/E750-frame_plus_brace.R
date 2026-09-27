".fac" <- function(x) {

  # exact factorial; NB returns (by design) a
  # bigz; used in setparts()
  return(prod(seq_along(x)))
}
}