getRanEf.CREM <- function(x, ...){

  out <- list(ranef_b = x$Random_Coefficients$ranef_b,
              ranef_g = x$Random_Coefficients$ranef_g)
  class(out) <- c("getRanEf.CREM", class(out))
  return(out)

}