#' @title Convert Phangorn reconstructions to a matrix
#' @description Converts a Phangorn reconstruction to a matrix of the most likely
#'   state at each site. If \code{best = TRUE}, the most likely state is chosen
#'   randomly in the event of a tie.
#' @param x A Phangorn reconstruction object.
#' @param site Which site to extract.
#' @param best If \code{TRUE}, the most likely state is chosen randomly in the event of a tie.
#' @return A matrix of the most likely state at each site.
#' @export
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