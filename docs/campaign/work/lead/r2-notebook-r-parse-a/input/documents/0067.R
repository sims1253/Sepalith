f.pval <- function(fval, df1, df2, ROPE = NULL) {
  m <- df1 + df2
  p <- if (is.null(ROPE)) {
    stats::pf(fval, df1, df2, lower.tail = FALSE)
  } else {
    stats::pf(fval, df1, df2, ncp = m * ROPE, lower.tail = T)
  }

  return(p)
}