setCov.UN <- function(Cov, traits, j, mo, saveAt) {
  message("UNstructured covariance matrix")

  if (is.null(Cov$df0)) {
    Cov$df0 <- traits + 1
    message("df0 was set to ", Cov$df0)
  }

  if (is.null(Cov$S0)) {
    #Cov$S0<-diag(traits)
    #message("S0 set to an identity matrix")
    Cov$S0 <- mo * (Cov$df0 + traits + 1)
    message("S0 set to ")
    print(Cov$S0)
  }

  #Omega=cov(b)
  Cov$Omega <- riwish(v = Cov$df0, S = Cov$S0)
  Cov$Omegainv <- solve(Cov$Omega)

  #Objects for saving posterior means for MCMC
  Cov$post_Omega <- matrix(0, nrow = traits, ncol = traits)
  Cov$post_Omega2 <- matrix(0, nrow = traits, ncol = traits)

  #Output files
  Cov$fName_Omega <- paste(saveAt, "Omega_", j, ".dat", sep = "")
  Cov$f_Omega <- file(description = Cov$fName_Omega, open = "w")

  return(Cov)
}