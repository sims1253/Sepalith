pmean <- function(x, scale, shape) {
  y <- scale *
    pgpd(x, scale = scale, shape = shape) /
    (1 - shape) -
    x * (1 - pgpd(x, scale = scale, shape = shape)) / (1 - shape)
  return(y)
}