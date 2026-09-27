bayes_mh_censored <- function(log_posterior, init_par, n_sim = 5000, burn_in = 1000, proposal_sd = 0.1) {
  p_dim <- length(init_par)
  if (length(proposal_sd) == 1) proposal_sd <- rep(proposal_sd, p_dim)
  
  chain <- matrix(0, nrow = n_sim, ncol = p_dim)
  chain[1, ] <- init_par
  curr_par <- init_par
  curr_lp <- log_posterior(curr_par)
  
  n_accept <- 0
  for (i in 2:n_sim) {
    cand <- curr_par + stats::rnorm(p_dim, mean = 0, sd = proposal_sd)
    cand_lp <- log_posterior(cand)
    if (!is.na(cand_lp) && !is.nan(cand_lp) && log(stats::runif(1)) < (cand_lp - curr_lp)) {
      curr_par <- cand
      curr_lp <- cand_lp
      n_accept <- n_accept + 1
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
    acceptance_rate = n_accept / n_sim,
    method = "Metropolis-Hastings M-H Algorithm"
  )
  class(res) <- "bayes_fit"
  return(res)
}