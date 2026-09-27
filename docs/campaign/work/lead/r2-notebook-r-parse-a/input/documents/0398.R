C2.DecisionLimit <- function(z, mu.C, R.C, A, x.var, alpha) {
  # Returns conditional decision limit for z, given x already in model
  #     Reference: Proposition Distribution of a C^2 contribution
  #                from Paper Criticality Assessment for
  #                Enhanced Multivariate Process Monitoring
  #
  #     Author: Dr. Víctor G. Tercero-Gómez, Dr. Diana Barraza-Barraza,
  #             Dr. A. Eduardo Cordero-Franco, Dr. Burcu Aytaçoğlu
  #
  #
  #     Date: October 6, 2023
  #     Versión: 1.0
  #
  # z :     observation vector, kx1, where z[x.var, ] correspond to variables already in the model
  # A:      list containing matrix decomposition of A, preferably, obtained from function
  #         decomposeA
  # R.C:    scalar, conditional covariance for z given x,
  # mu.C :  scalar, conditional mean for z given x
  # x.var:  vector indicating variables already present in the model. length: k-1
  # alpha : confidence level for decision limit

  # change of variables
  x <- z[x.var, ]
  lambda <- (R.C / (A$A2zx)) # Eq. 49
  m <- (mu.C / sqrt(R.C) - ((t(A$Azx) %*% solve(A$Axx)) %*% x) / sqrt(R.C)) # Eq. 49
  nc <- m^2

  conditionalCL <- lambda * qchisq(p = alpha, df = 1, ncp = as.numeric(nc))
  return(conditionalCL)
}