importance_sampling_censored <- function(log_target, proposal_pdf, rproposal, n_sim = 10000) {
  draws <- rproposal(n_sim)
  if (is.vector(draws)) draws <- matrix(draws, ncol = 1)
  
  log_w <- apply(draws, 1, log_target) - log(apply(draws, 1, proposal_pdf))
  max_lw <- max(log_w, na.rm = TRUE)
  w <- exp(log_w - max_lw)
  w[is.na(w) | is.nan(w)] <- 0
  w_norm <- w / sum(w)
  
  est_mean <- colSums(draws * w_norm)
  ess <- 1 / sum(w_norm^2)
  
  res <- list(
    draws = draws,
    weights = w_norm,
    post_means = est_mean,
    effective_sample_size = ess,
    method = "Importance Sampling"
  )
  class(res) <- "bayes_fit"
  return(res)
}