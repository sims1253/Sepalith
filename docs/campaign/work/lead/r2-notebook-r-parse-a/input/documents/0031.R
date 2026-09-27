make_dist <- function(
  x,
  Fx,
  sorted = FALSE,
  rearrange = FALSE,
  force01 = FALSE,
  method = "constant"
) {
  if (!sorted) {
    tmat <- cbind(x, Fx)
    tmat <- tmat[order(x), , drop = FALSE]
    x <- tmat[, 1]
    Fx <- tmat[, 2]
  }

  if (force01) {
    Fx <- sapply(Fx, function(Fxval) max(min(Fxval, 1), 0))
  }

  if (rearrange) {
    Fx <- sort(Fx)
  }

  retF <- approxfun(
    x,
    Fx,
    method = method,
    yleft = 0,
    yright = 1,
    f = 0,
    ties = "ordered"
  )
  class(retF) <- c("ecdf", "stepfun", class(retF))
  assign("nobs", length(x), envir = environment(retF))
  retF
}