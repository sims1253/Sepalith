reedge <- function(x1, y1, x2, y2, r1, r2) {
  d <- sqrt((x1 - x2)^2 + (y1 - y2)^2)
  st <- -(y2 - y1) / d
  ct <- (x2 - x1) / d
  A <- matrix(c(ct, -st, st, ct), 2)

  M <- c(x1, y1)
  p1 <- c(r1, 0)
  a1 <- A %*% p1 + M
  a1 <- as.vector(a1)
  p2 <- c(d - r2, 0)
  a2 <- A %*% p2 + M
  a2 <- as.vector(a2)

  return(c(a1, a2))
}