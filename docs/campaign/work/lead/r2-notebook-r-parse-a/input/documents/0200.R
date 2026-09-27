gammaParamsConvert <- function(...) {
  l <- list(...)

  if (length(l) != 2) {
    stop("Number of input arguments should be two.")
  }

  if (is.null(l$mean)) {
    l$mean <- l$shape * l$scale
  }

  if (is.null(l$sd)) {
    l$sd <- sqrt(l$shape * l$scale^2)
  }

  if (is.null(l$shape)) {
    l$shape <- (l$mean / l$sd)^2
  }

  if (is.null(l$scale)) {
    l$scale <- l$sd^2 / l$mean
  }

  return(l)
}