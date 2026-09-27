.fiml_loglik <- function(data, mu, sigma, patterns = NULL) {
  # A non-positive-definite sigma has no valid MVN density; refuse with a classed error rather
  # than letting chol() raise a bare base error (this helper is reused with externally supplied
  # moments). A principal submatrix of a positive-definite matrix is itself positive definite,
  # so this single check covers Sigma_oo for every pattern. The threshold matches the
  # non-positive-definite guard in .prepare_cor_input().
  if (
    any(
      eigen(sigma, symmetric = TRUE, only.values = TRUE)$values <
        .Machine$double.eps
    )
  ) {
    cli::cli_abort(
      c(
        "The covariance matrix is not positive definite, so the FIML log-likelihood is undefined.",
        "i" = "A variable may be (near-)constant or collinear."
      ),
      class = "efa_fiml_not_posdef"
    )
  }

  Y <- as.matrix(data)

  # The EM engine already grouped the rows it passes (over its fully-missing-row-filtered data),
  # so accept those patterns directly; otherwise group here so the helper stays self-contained
  # for external callers that pass raw, unfiltered data (fully-missing rows carry no density).
  if (is.null(patterns)) {
    obs <- !is.na(Y)
    keep <- rowSums(obs) > 0L
    Y <- Y[keep, , drop = FALSE]
    patterns <- .fiml_patterns(obs[keep, , drop = FALSE])
  }

  log2pi <- log(2 * pi)
  ll <- 0
  for (pat in patterns) {
    o <- pat$o
    Yo <- Y[pat$rows, o, drop = FALSE]
    ch <- chol(sigma[o, o, drop = FALSE])
    logdet <- 2 * sum(log(diag(ch))) # log|Sigma_oo|
    C <- sweep(Yo, 2L, mu[o], "-")
    quad <- rowSums((C %*% chol2inv(ch)) * C) # (y_o - mu_o)' Sigma_oo^{-1} (.)
    ll <- ll + sum(-0.5 * (length(o) * log2pi + logdet + quad))
  }
  ll
}