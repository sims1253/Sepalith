car_mm <- function(d_list, ...) {
  out <- rstan::sampling(stanmodels$CARMM_COV_P, data = d_list, ...)
  return(out)
}