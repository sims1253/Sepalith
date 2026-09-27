`dimnames<-.BEDMatrix` <- function(x, value) {
  d <- dim(x)
  v1 <- value[[1L]]
  v2 <- value[[2L]]
  if (
    !is.list(value) ||
      length(value) != 2L ||
      !(is.null(v1) || length(v1) == d[1L]) ||
      !(is.null(v2) || length(v2) == d[2L])
  ) {
    stop("invalid dimnames", call. = FALSE)
  }
  slot(x, "dnames") <- lapply(value, function(v) {
    if (!is.null(v)) {
      as.character(v)
    }
  })
  return(x)
}