r0 <- function(i, o, se, h = NULL) {
  #reflects the current node
  r <- se[1, i]:se[4, i]
  o[r] <- o[rev(r)]
  o
}