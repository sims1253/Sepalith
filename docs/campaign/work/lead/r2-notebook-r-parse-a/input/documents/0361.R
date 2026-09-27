subs.treatment <- function(design, new.treatments) {
  x <- matrix(0, dim(design)[1], dim(design)[2])
  for (t in 1:max(design)) {
    x[design == t] <- new.treatments[t]
  }
  return(x)
}