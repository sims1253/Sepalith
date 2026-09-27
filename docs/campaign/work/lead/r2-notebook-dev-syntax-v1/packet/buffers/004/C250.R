#' @title b.rcs
#'
#' @description Cubic spline interpolation of \code{x} at knots \code{knots}
#'
#' @param x numeric vector of x values
#' @param knots numeric vector of knots
#' @param inclx logical, if \code{TRUE} then x is included in output
#'
#' @return numeric matrix of spline values
#'
#' @export
#'
b_rcs <- function(x, knots, inclx = FALSE) {
  num.knots <- length(knots)
  tk <- knots[num.knots]
  tkmin1 <- knots[num.knots - 1]

  res <- lapply(1:(num.knots - 2), function(i) {
    tj <- knots[i]
    pmax((x - tj)^3, 0) -
      pmax((x - tkmin1)^3, 0) * (tk - tj) / (tk - tkmin1) +
      pmax((x - tk)^3, 0) * (tkmin1 - tj) / (tk - tkmin1)
  })

  res <- matrix(unlist(res), ncol = num.knots - 2)

  if (inclx) {
    res <- cbind(x, res)
  }

  # return result
  return(res)
} # end b.rcs