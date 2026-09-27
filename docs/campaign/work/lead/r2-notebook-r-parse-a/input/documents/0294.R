getRanEf.BPREM <- function(x, ...){

  out <- list(ranef = x$Random_Coefficients$ranef_b)
  class(out) <- c("getRanEf.BPREM", class(out))
  return(out)

}