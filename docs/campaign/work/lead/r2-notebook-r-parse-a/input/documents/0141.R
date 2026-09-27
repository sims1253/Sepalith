gpdcostfun <- function(x, scale, shape, tau) {
  mu <- scale / (1 - shape)
  pmx <- pmean(x, scale, shape)
  pgx <- pgpd(x, scale = scale, shape = shape)

  ppx <- pmx - x * pgx
  res <- ppx / (2 * ppx + (x - mu))

  return(res - tau)
}