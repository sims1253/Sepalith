get_priors <- function(num_priors) {
  # Get uniform priors
  prior_vec <- rep(1 / num_priors, num_priors)
  return(prior_vec)
}