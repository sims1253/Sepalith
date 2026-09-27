".fac" <- function(x) {

  # exact factorial; NB returns (by design) a
  # bigz; used in setparts()
  out <- as.bigz(1)
  for (n in x) {
    out <- out * prod(as.bigz(seq_len(n)))
  }
  return(out)
}