guessDistMethod <- function(dist.method, map) {
  if (is.null(dist.method)) {
    dist.method <- 'euclidean'
    if (!is.null(map$distf)) {
      if (map$distf == 1) {
        dist.method <- 'manhattan'
      }
      if (map$distf == 3) dist.method <- 'chebyshev'
    }
  }
  dist.method
}