sqrt1pSqr <- function(z) {
  if (!length(z)) {
    return(z)
  }
  z2 <- z^2
  u <- 1 + z2
  r <- sqrt(u) # direct form for normal case
  if (length(sml <- which(1 - z2^2 / 8 == 1))) {
    # also "works" for mpfr
    z22 <- z2[sml] / 2
    r[sml] <- 1 + z22 * (1 - z22 / 2) # 2-term approx 1 + z^2/2 - z^4/8
  }
  r
}