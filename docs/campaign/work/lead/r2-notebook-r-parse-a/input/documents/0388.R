vuong.norm <- function(
  qcond,
  tau2par,
  param1,
  param2,
  TP,
  FN,
  FP,
  TN,
  gl,
  mgrid
) {
  prob1 <- jpmf.norm(param1, TP, FN, FP, TN, gl, mgrid, qcondbvn, tau2par.bvn)
  n <- length(prob1)
  prob2 <- jpmf.norm(param2, TP, FN, FP, TN, gl, mgrid, qcond, tau2par)
  m <- log(prob2 / prob1)
  z <- sqrt(n) * mean(m) / sd(m)
  pvalue <- 2 * pnorm(-abs(z))
  result <- data.frame(round(z, digits = 3), round(pvalue, digits = 3))
  names(result) <- c("z", "p.value")
  return(result)
}