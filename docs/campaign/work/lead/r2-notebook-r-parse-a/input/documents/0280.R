sort.BFodds <- function(x, decreasing = FALSE, ...) {
  ord <- order(extractOdds(x, logodds = TRUE)$odds, decreasing = decreasing)
  return(x[ord])
}