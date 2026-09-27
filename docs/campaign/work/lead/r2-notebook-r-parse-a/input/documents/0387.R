jpmf.beta <- function(param, TP, FN, FP, TN, gl, mgrid, qcond, tau2par) {
  p <- param[1:2]
  g <- param[3:4]
  tau <- param[5]
  u1 <- mgrid$x
  th <- tau2par(tau)
  u2 <- qcond(mgrid$y, mgrid$x, th)
  a <- p / g - p
  b <- (1 - p) * (1 - g) / g
  x1 <- qbeta(u1, a[1], b[1])
  x2 <- qbeta(u2, a[2], b[2])
  N <- length(TP)
  prob <- rep(NA, N)
  for (i in 1:N) {
    temp <- binomprod(x1, x2, TP[i], FN[i], FP[i], TN[i])
    prob[i] <- gl$w %*% temp %*% as.matrix(gl$w)
  }
  prob
}