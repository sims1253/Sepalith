# Return zero for an empty vector, otherwise sum its positive values.
sum_positive <- function(x) {
  if (length(x) == 0) {
    return(0)
  }
  sum(x[x > 0])


