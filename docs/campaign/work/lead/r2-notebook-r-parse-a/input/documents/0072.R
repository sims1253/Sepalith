.mc_normalize_prob <- function(p, eps = 1e-12) {
  p <- as.numeric(p)

  if (!length(p)) {
    return(p)
  }

  p[!is.finite(p) | p < 0] <- 0

  if (sum(p) <= eps) {
    p[] <- 1 / length(p)
  } else {
    p <- p / sum(p)
  }

  p
}