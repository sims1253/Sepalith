guessDim <- function(dim, map) {
  if (is.null(dim)) {
    if (is.null(map$grid)) {
      dim <- 2
    } else {
      dim <- ncol(map$grid)
    }
  }
  dim
}