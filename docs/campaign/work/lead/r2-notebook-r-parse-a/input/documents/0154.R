GaussSmoothKernel <- function(
  voxdim = c(1, 1, 1),
  ksize = 5,
  sigma = diag(3, 3)
) {
  #calculates a discretized smoothing kernel in up to 3 dimensions given an arbitrary covariance matrix
  #sigma is covariance matrix of the gaussian
  #doesn't have to be non-singular; zero on the diagonal of sigma indicate no smoothing in that direction

  if ((2 * floor(ksize / 2)) == ksize) {
    stop(paste("ksize must be odd"))
  }

  a <- array(0, dim = c(ksize, ksize, ksize))
  centre <- (ksize + 1) / 2

  sig.ck <- c(TRUE, TRUE, TRUE)
  for (i in 1:3) {
    if (sigma[i, i] == 0) {
      sigma[i, i] <- 1
      sig.ck[i] <- FALSE
    }
  }
  sig.inv <- solve(sigma)
  sig.det <- abs(det(sigma))

  for (i in 1:ksize) {
    for (j in 1:ksize) {
      for (k in 1:ksize) {
        x <- (c(i, j, k) - centre) * voxdim
        a[i, j, k] <- ((2 * pi)^(-3 / 2)) *
          exp(-0.5 * (t(x) %*% sig.inv %*% x)) /
          sqrt(sig.det)
      }
    }
  }
  if (!sig.ck[1]) {
    a[-centre, , ] <- 0
  }
  if (!sig.ck[2]) {
    a[, -centre, ] <- 0
  }
  if (!sig.ck[3]) {
    a[,, -centre] <- 0
  }
  a <- a / sum(a)

  return(a)
}