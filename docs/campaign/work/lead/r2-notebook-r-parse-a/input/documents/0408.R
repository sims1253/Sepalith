.simulation_smoother <- function(y, W, m0, P0, Q, r, a_sm_real, n_draws, seed) {
  set.seed(seed)
  Tn <- length(y)
  K <- ncol(W)
  cP0 <- chol(P0)
  cQ <- chol(Q)
  out <- array(NA_real_, dim = c(Tn, K, n_draws))
  for (d in seq_len(n_draws)) {
    phi_plus <- matrix(NA_real_, Tn, K)
    phi_plus[1, ] <- m0 + as.numeric(crossprod(cP0, stats::rnorm(K)))
    for (t in 2:Tn) {
      phi_plus[t, ] <- phi_plus[t - 1, ] +
        as.numeric(crossprod(cQ, stats::rnorm(K)))
    }
    y_plus <- vapply(
      seq_len(Tn),
      function(t) {
        sum(W[t, ] * phi_plus[t, ]) + stats::rnorm(1, 0, sqrt(r))
      },
      numeric(1)
    )
    sm_plus <- .kalman_rts(y_plus, W, m0, P0, Q, r)
    out[,, d] <- phi_plus - sm_plus$mean + a_sm_real
  }
  out
}