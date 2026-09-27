wPB.AM <- function(
  test.results,
  weights,
  alpha = 0.05,
  zeta = 0.5,
  critical.values = FALSE,
  exact = TRUE,
  select.threshold = 1
) {
  out <- weighted.PB(
    test.results,
    weights,
    alpha,
    zeta,
    "AM",
    critical.values,
    exact,
    select.threshold
  )

  out$Data$Data.name <- paste(
    deparse(substitute(test.results)),
    "and",
    deparse(substitute(weights))
  )

  return(out)
}