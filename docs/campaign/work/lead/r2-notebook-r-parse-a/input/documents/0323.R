weighted.PB <- function(
  test.results,
  weights = NULL,
  alpha = 0.05,
  zeta = 0.5,
  weighting.method = c("AM", "GM"),
  critical.values = FALSE,
  exact = TRUE,
  select.threshold = 1
) {
  #----------------------------------------------------
  #       check arguments
  #----------------------------------------------------
  # test results (p-values)
  assert(
    check_numeric(
      x = test.results,
      lower = 0,
      upper = 1,
      any.missing = FALSE,
      min.len = 1
    ),
    check_r6(
      x = test.results,
      classes = "DiscreteTestResults",
      public = c("get_pvalues", "get_pvalue_supports", "get_support_indices")
    )
  )
  pvals <- if (is.numeric(test.results)) {
    test.results
  } else {
    test.results$get_pvalues()
  }
  n <- length(pvals)

  # weights
  assert_numeric(
    x = weights,
    lower = 0,
    finite = TRUE,
    len = n,
    all.missing = FALSE,
    null.ok = TRUE
  )
  if (is.null(weights)) {
    weights <- rep(1, n)
  }

  # FDP level
  qassert(x = alpha, rules = "N1[0, 1]")

  # Exceedance probability
  qassert(x = zeta, rules = "N1[0, 1]")

  # Weighting method
  qassert(x = weighting.method, rules = "S1")
  match.arg(toupper(weighting.method), c("AM", "GM"))

  # compute and return critical values?
  qassert(critical.values, "B1")

  # exact computation of Poisson-Binomial distribution or normal approximation?
  qassert(exact, "B1")

  # selection threshold
  qassert(x = select.threshold, rules = "N1(0, 1]")

  #----------------------------------------------------
  #       execute computations
  #----------------------------------------------------
  output <- weighted.fdx.int(
    pvec = pvals,
    weights = weights,
    method = "PB",
    weight.meth = weighting.method,
    alpha = alpha,
    zeta = zeta,
    exact = exact,
    crit.consts = critical.values,
    threshold = select.threshold,
    data.name = paste(
      deparse(substitute(test.results)),
      "and",
      deparse(substitute(weights))
    )
  )

  return(output)
}