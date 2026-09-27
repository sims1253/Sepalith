#' Convert a Phangorn reconstruction to a matrix
#'
#' @param x A Phangorn reconstruction
#' @param site Which site to convert
#' @param best If TRUE, return the best reconstruction for each site
#'
#' @return A matrix of the reconstruction
#'
#' @noRd
ConvertPhangornReconstructions <- function(x, site = 1, best = TRUE) {
  x.local <- subset(x, , site)
  nc <- attr(x.local, "nc")
  y <- matrix(unlist(x.local[]), ncol = nc, byrow = TRUE)
  rownames(y) <- names(x.local[])
  colnames(y) <- attr(x.local, "levels")
  result <- y
  if (best) {
    best.vector <- rep(NA, nrow(result))
    for (i in sequence(nrow(result))) {
      best.vector[i] <- sample(which.max(result[i, ]), 1) #so resolve ties randomly
    }
    names(best.vector) <- rownames(y)
    result <- best.vector
  }
  return(result)
}