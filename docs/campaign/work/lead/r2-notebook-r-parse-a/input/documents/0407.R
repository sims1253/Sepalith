disaggregate_conjugate <- function(
  cpi,
  W,
  years = NULL,
  industries = NULL,
  q_frac = 0.10,
  r_frac = 0.05,
  p0_frac = 0.30,
  n_draws = 0L,
  seed = 1234L
) {
  cpi <- as.numeric(cpi)
  if (length(cpi) < 2L || !all(is.finite(cpi)) || any(cpi <= 0)) {
    stop(
      "`cpi` must be a finite, strictly positive numeric vector of length >= 2.",
      call. = FALSE
    )
  }
  W <- row_norm1(as.matrix(W))
  Tn <- length(cpi)
  K <- ncol(W)
  if (nrow(W) != Tn) {
    stop("`W` rows must match length(cpi).", call. = FALSE)
  }

  industries <- industries %||% colnames(W) %||% paste0("sector_", seq_len(K))
  years <- years %||% seq_len(Tn)

  sd_cpi <- stats::sd(cpi)
  r <- max(.Machine$double.eps, (r_frac * sd_cpi)^2)
  Q <- diag((q_frac * sd_cpi)^2, K)
  m0 <- rep(cpi[1], K)
  P0 <- diag((p0_frac * cpi[1])^2, K)

  sm <- .kalman_rts(cpi, W, m0, P0, Q, r)
  med <- sm$mean
  sdv <- sqrt(sm$var)
  lo <- med - 1.959964 * sdv
  hi <- med + 1.959964 * sdv
  dimnames(med) <- dimnames(lo) <- dimnames(hi) <- list(years, industries)

  agg_mean <- rowSums(W * med)
  agg_var <- vapply(
    seq_len(Tn),
    function(t) {
      as.numeric(crossprod(W[t, ], sm$P_sm[[t]] %*% W[t, ]))
    },
    numeric(1)
  )
  agg_summary <- cbind(
    q2.5 = agg_mean - 1.959964 * sqrt(agg_var),
    median = agg_mean,
    q97.5 = agg_mean + 1.959964 * sqrt(agg_var)
  )
  rownames(agg_summary) <- years

  # Gaussian total log-likelihood of the aggregate at the smoothed fit.
  loglik <- sum(stats::dnorm(cpi, agg_mean, sqrt(agg_var + r), log = TRUE))

  phi_draws <- NULL
  n_draws <- as.integer(n_draws)
  if (n_draws > 0L) {
    phi_draws <- .simulation_smoother(
      cpi,
      W,
      m0,
      P0,
      Q,
      r,
      sm$mean,
      n_draws,
      seed
    )
    dimnames(phi_draws) <- list(years, industries, NULL)
  }

  out <- list(
    phi_summary = list(median = med, q2.5 = lo, q97.5 = hi),
    agg_summary = agg_summary,
    loglik = loglik,
    phi_draws = phi_draws,
    cpi = cpi,
    W = W,
    years = years,
    industries = industries,
    config = list(
      T = Tn,
      K = K,
      q_frac = q_frac,
      r_frac = r_frac,
      p0_frac = p0_frac,
      n_draws = n_draws
    )
  )
  class(out) <- c("disagg_conjugate", "list")
  out
}