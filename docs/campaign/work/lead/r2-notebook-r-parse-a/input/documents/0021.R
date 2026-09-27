format1 <- function(expr) {
  res <- if (is.symbol(expr)) {
    as.character(expr)
  } else if (is.call(expr) && expr[[1]] == as.symbol("{")) {
    c(sapply(as.list(expr), format1), "}")
  } else {
    format(expr)
  }
  n <- length(res)
  if (n > 1) {
    if (endsWith(res[1], "{") && n > 2) {
      b <- paste0(res[-1], collapse = "; ")
      res <- paste0(res[1], b, collapse = "")
    } else {
      res <- paste0(res, collapse = " ")
    }
  }
  return(res)
}