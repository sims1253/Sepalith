predict.fk_regression <- function(object, xtest = NULL, ...) {
  if (is.null(xtest)) {
    xtest <- object$x
  }

  mn <- mean(object$x)

  xtest <- xtest - mn
  x <- object$x - mn

  o <- order(x)

  otest <- order(xtest)

  if (object$type == 'loc-lin') {
    sK <- ksum(
      x[o],
      numeric(length(x)) + 1,
      xtest[otest],
      object$h,
      object$betas
    )
    sKx <- ksum(x[o], x[o], xtest[otest], object$h, object$betas)
    sKy <- ksum(x[o], object$y[o], xtest[otest], object$h, object$betas)
    sKx2 <- ksum(x[o], x[o]^2, xtest[otest], object$h, object$betas)
    sKxy <- ksum(x[o], x[o] * object$y[o], xtest[otest], object$h, object$betas)
    yhat <- (((sKx2 * sKy - sKx * sKxy) +
      (sK * sKxy - sKx * sKy) * xtest[otest]) /
      (sK * sKx2 - sKx^2))[rank(xtest, ties.method = "first")]
  } else {
    sK <- ksum(
      x[o],
      numeric(length(x)) + 1,
      xtest[otest],
      object$h,
      object$betas
    )
    sKy <- ksum(x[o], object$y[o], xtest[otest], object$h, object$betas)
    yhat <- (sKy / sK)[rank(xtest, ties.method = "first")]
  }
  yhat
}