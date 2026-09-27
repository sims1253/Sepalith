max.BFodds <- function(..., na.rm = FALSE) {
  joinedodds <- do.call('c', list(...))
  el <- head(joinedodds, n = 1)
  return(el)
}