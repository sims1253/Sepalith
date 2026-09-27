# Return the mean of finite values; return NA_real_ when none remain.
mean_finite <- function(x) {
  finite <- is.finite(x)
  if (!any(finite)) {
    return(NA_real_)
  }
  mean(x[finite])


