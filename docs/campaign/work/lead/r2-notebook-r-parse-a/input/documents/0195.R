summary.svy_vglm <- function(object, ...) {
  object$coeftable <- cbind(
    Coef = object$coef,
    SE = sqrt(diag(object$var)),
    z = object$coef / sqrt(diag(object$var)),
    p = pnorm(-abs(object$coef / sqrt(diag(object$var)))) * 2
  )
  class(object) <- "summary.svy_vglm"
  object
}