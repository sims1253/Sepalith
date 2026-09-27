t0 <- function(i, o, se, h = NULL) {
  #translates the current node
  r <- se[1, i]:se[4, i]
  r1 <- c(se[3, i]:se[4, i], se[1, i]:se[2, i])
  o[r] <- o[r1]

  o
}