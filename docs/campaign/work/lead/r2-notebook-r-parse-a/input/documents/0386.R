jpmf.norm <- function(param, TP, FN, FP, TN, gl, mgrid, qcond, tau2par) {
  p <- param[1:2]
  si <- param[3:4]
  tau <- param[5]
  mu <- log(p / (1 - p))
  u1 <- mgrid$x
  th <- tau2par(tau)
  u2 <- qcond(mgrid$y, mgrid$x, th)
  x1 <- qnorm(u1, mu[1], si[1])
  x2 <- qnorm(u2, mu[2], si[2])
  t1 <- exp(x1)
  t2 <- exp(x2)
  x1 <- t1 / (1 + t1)
  x2 <- t2 / (1 + t2)
  N <- length(TP)
  prob <- rep(NA, N)
  for (i in 1:N) {
    temp <- binomprod(x1, x2, TP[i], FN[i], FP[i], TN[i])
    prob[i] <- gl$w %*% temp %*% as.matrix(gl$w)
  }
  prob
}