ss.optim <- function(delta, R, alpha) {
  # Initial proxy weights (log-transformed)
  nvar <- dim(R)[1]
  w0 <- rep(1, nvar)
  theta0 <- log(w0)

  # Inner function to compute total out-of-control ARL
  funARLs <- function(theta, delta, alpha, R) {
    # Transform proxies to positive weights normalized to sum to 10
    # (scaling to avoid numerical issues)
    w_pos <- exp(theta)
    w_full <- 10 * w_pos / sum(w_pos)

    # Control limit calculation
    h <- wChisq.CLim(w = w_full, R = R, alpha = alpha)$Control_Limit

    # Sum ARLs for each shift vector
    ARLs <- 0
    for (i in seq_along(delta)) {
      ARLs <- ARLs +
        wChisq.arl(delta = delta[[i]], R = R, h = h, w = w_full)$arl
    }
    return(ARLs)
  }

  # Optimize proxy weights using quasi-Newton method
  opt_result <- optim(
    par = theta0,
    fn = funARLs,
    delta = delta,
    alpha = alpha,
    R = R,
    method = "BFGS"
  )

  # Convert optimized proxies to normalized weights (sum to 1)
  w_opt <- exp(opt_result$par) / sum(exp(opt_result$par))

  # Compute final control limit
  h <- wChisq.CLim(w = w_opt, R = R, alpha = alpha)

  results <- list(
    weights = w_opt,
    control_limit = h$Control_Limit,
    objective = opt_result$value,
    method = "BFGS",
    n_starts = 1
  )
  return(results)
}