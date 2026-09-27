get.R <- function(Sigma0) {
  # Returns R matrix from variance-covariance matrix Sigma0
  #     Reference: Eq. (16) from Paper Criticality Assessment for
  #                Enhanced Multivariate Process Monitoring
  #     Author: Dr. Víctor G. Tercero-Gómez, Dr. Diana Barraza-Barraza,
  #             Dr. A. Eduardo Cordero-Franco, Dr. Burcu Aytaçoğlu
  #
  #     Date: October 6, 2023
  #     Versión: 1.0

  sigma0.dim <- dim(Sigma0)
  dim.rows <- sigma0.dim[1]
  dim.cols <- sigma0.dim[2]

  R <- matrix(rep(NA, dim.rows * dim.cols), ncol = dim.cols)
  for (ro in 1:dim.rows) {
    for (co in 1:dim.cols) {
      R[ro, co] <- Sigma0[ro, co] /
        sqrt(as.numeric(diag(Sigma0))[ro]) /
        sqrt(as.numeric(diag(Sigma0))[co])
    }
  }
  return(R)
}