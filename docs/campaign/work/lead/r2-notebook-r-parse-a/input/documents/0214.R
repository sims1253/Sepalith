.OrderAlphabetically <- function(counts) {
  ## Orders the 'table' object counts alphabetically by name.
  index <- order(names(counts))
  return(counts[index])
}