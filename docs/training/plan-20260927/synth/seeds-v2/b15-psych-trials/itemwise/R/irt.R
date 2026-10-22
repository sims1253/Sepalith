#' Two-parameter logistic response probability
#'
#' Probability of a keyed response under the 2PL model for every combination
#' of ability and item.
#'
#' @param theta A numeric vector of person abilities.
#' @param a A numeric vector of item discriminations.
#' @param b A numeric vector of item difficulties.
#' @param D Scaling constant; use 1.702 to approximate the normal ogive.
#' @return A matrix with one row per `theta` and one column per item.
#' @export
irt_prob_2pl = function(theta, a, b, D = 1) {
  stopifnot(length(a) == length(b))
  z = outer(theta, seq_along(a), function(t, j) D * a[j] * (t - b[j]))
  stats::plogis(z)
}

#' Three-parameter logistic response probability
#'
#' Probability of a keyed response under the 3PL model, which adds a lower
#' asymptote for guessing to the 2PL.
#'
#' @param theta A numeric vector of person abilities.
#' @param a A numeric vector of item discriminations.
#' @param b A numeric vector of item difficulties.
#' @param c A numeric vector of item lower asymptotes.
#' @param D Scaling constant; use 1.702 to approximate the normal ogive.
#' @return A matrix with one row per `theta` and one column per item.
#' @export
irt_prob_3pl = function(theta, a, b, c, D = 1) {
  stopifnot(length(a) == length(b), length(b) == length(c))
  p = irt_prob_2pl(theta, a, b, D = D)
  guess = matrix(c, nrow = length(theta), ncol = length(c), byrow = TRUE)
  guess + (1 - guess) * p
}

#' Item information
#'
#' Fisher information each item contributes at each ability value, under the
#' 2PL or, allowing for a non-zero lower asymptote, the 3PL model.
#'
#' @param theta A numeric vector of person abilities.
#' @param a A numeric vector of item discriminations.
#' @param b A numeric vector of item difficulties.
#' @param D Scaling constant; use 1.702 to approximate the normal ogive.
#' @return A matrix with one row per `theta` and one column per item.
#' @rdname item_info
#' @export
item_info_2pl = function(theta, a, b, D = 1) {
  p = irt_prob_2pl(theta, a, b, D = D)
  a_mat = matrix(a, nrow = length(theta), ncol = length(a), byrow = TRUE)
  (D * a_mat)^2 * p * (1 - p)
}

#' @param c A numeric vector of item lower asymptotes.
#' @rdname item_info
#' @export
item_info_3pl = function(theta, a, b, c, D = 1) {
  p = irt_prob_3pl(theta, a, b, c, D = D)
  a_mat = matrix(a, nrow = length(theta), ncol = length(a), byrow = TRUE)
  c_mat = matrix(c, nrow = length(theta), ncol = length(c), byrow = TRUE)

  # Birnbaum's formula; reduces to the 2PL information when c = 0
  (D * a_mat)^2 * ((1 - p) / p) * ((p - c_mat) / (1 - c_mat))^2
}

#' Test information function
#'
#' Sums item information across items for a parameter table.
#'
#' @param theta A numeric vector of person abilities.
#' @param pars A data frame of item parameters with columns `a`, `b` and,
#'   for the 3PL, `c`.
#' @param model The IRT model, `"2pl"` or `"3pl"`.
#' @param D Scaling constant; use 1.702 to approximate the normal ogive.
#' @return A numeric vector of test information, one value per `theta`.
#' @export
test_information = function(theta, pars, model = c("2pl", "3pl"), D = 1) {
  model = match.arg(model)
  info = switch(model,
    "2pl" = item_info_2pl(theta, pars$a, pars$b, D = D),
    "3pl" = item_info_3pl(theta, pars$a, pars$b, pars$c, D = D)
  )
  rowSums(info)
}

#' Conditional standard error of measurement
#'
#' The reciprocal square root of the test information at each ability value.
#'
#' @inheritParams test_information
#' @return A numeric vector of standard errors, one value per `theta`.
#' @export
theta_se = function(theta, pars, model = c("2pl", "3pl"), D = 1) {
  info = test_information(theta, pars, model = model, D = D)
  1 / sqrt(info)
}

#' Expected a posteriori ability estimates
#'
#' Approximates each respondent's posterior mean ability on a fixed
#' quadrature grid under a normal prior.
#'
#' @param responses A 0/1 matrix of item responses, one row per respondent.
#' @param pars A data frame of item parameters with columns `a`, `b` and,
#'   for the 3PL, `c`.
#' @param model The IRT model, `"2pl"` or `"3pl"`.
#' @param n_nodes Number of quadrature nodes.
#' @param prior_mean,prior_sd Mean and standard deviation of the normal prior.
#' @return A data frame with columns `theta` and `se`, one row per respondent.
#' @export
estimate_theta_eap = function(responses, pars, model = c("2pl", "3pl"),
                              n_nodes = 61, prior_mean = 0, prior_sd = 1) {
  model = match.arg(model)
  responses = as.matrix(responses)
  if (ncol(responses) != nrow(pars)) {
    stop("`responses` needs one column per row of `pars`.", call. = FALSE)
  }

  nodes = seq(prior_mean - 4 * prior_sd, prior_mean + 4 * prior_sd, length.out = n_nodes)
  prior = stats::dnorm(nodes, prior_mean, prior_sd)
  p = switch(model,
    "2pl" = irt_prob_2pl(nodes, pars$a, pars$b),
    "3pl" = irt_prob_3pl(nodes, pars$a, pars$b, pars$c)
  )
  # Keep the logs finite: with steep items p can round to exactly 0 or 1,
  # and 0 * log(0) would turn the whole likelihood into NaN.
  p = pmin(pmax(p, 1e-12), 1 - 1e-12)
  log_p = log(p)
  log_q = log(1 - p)

  est = numeric(nrow(responses))
  se = numeric(nrow(responses))
  for (i in seq_len(nrow(responses))) {
    u = responses[i, ]
    seen = !is.na(u)
    # log-likelihood at every node, skipping unanswered items
    ll = log_p[, seen, drop = FALSE] %*% u[seen] +
      log_q[, seen, drop = FALSE] %*% (1 - u[seen])
    post = exp(ll - max(ll)) * prior
    post = post / sum(post)
    est[i] = sum(nodes * post)
    se[i] = sqrt(sum((nodes - est[i])^2 * post))
  }
  data.frame(theta = est, se = se)
}

#' Simulate dichotomous responses from item parameters
#'
#' Draws 0/1 responses for a set of abilities, useful for checking
#' estimation code and planning test length.
#'
#' @param theta A numeric vector of person abilities.
#' @param pars A data frame of item parameters with columns `a`, `b` and,
#'   for the 3PL, `c`.
#' @param model The IRT model, `"2pl"` or `"3pl"`.
#' @param seed Optional random seed.
#' @return An integer matrix with one row per `theta` and one column per item.
#' @export
simulate_responses = function(theta, pars, model = c("2pl", "3pl"), seed = NULL) {
  model = match.arg(model)
  if (!is.null(seed)) {
    set.seed(seed)
  }

  p = switch(model,
    "2pl" = irt_prob_2pl(theta, pars$a, pars$b),
    "3pl" = irt_prob_3pl(theta, pars$a, pars$b, pars$c)
  )
  u = matrix(stats::runif(length(p)), nrow = nrow(p)) < p
  storage.mode(u) = "integer"

  colnames(u) = if (is.null(pars$item)) paste0("i", seq_len(ncol(u))) else pars$item
  u
}
