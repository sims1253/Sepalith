bayes_gibbs_censored <- function(log_posterior, init_par, n_sim = 5000, burn_in = 1000) {
  p_dim <- length(init_par)
  chain <- matrix(0, nrow = n_sim, ncol = p_dim)
  chain[1, ] <- init_par
  
  curr_par <- init_par
  for (i in 2:n_sim) {
    for (j in 1:p_dim) {
      cand <- curr_par
      cand[j] <- stats::rnorm(1, mean = curr_par[j], sd = 0.1)
      if (cand[j] > 0) {
        log_alpha <- log_posterior(cand) - log_posterior(curr_par)
        if (!is.na(log_alpha) && log(stats::runif(1)) < log_alpha) {
          curr_par[j] <- cand[j]
        }
      }
    }
    chain[i, ] <- curr_par
  }
  
  post_chain <- chain[(burn_in + 1):n_sim, , drop = FALSE]
  post_means <- colMeans(post_chain)
  ci_lower <- apply(post_chain, 2, function(x) stats::quantile(x, 0.025))
  ci_upper <- apply(post_chain, 2, function(x) stats::quantile(x, 0.975))
  
  res <- list(
    chain = post_chain,
    post_means = post_means,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    method = "Gibbs Sampling"
  )
  class(res) <- "bayes_fit"
  return(res)
}