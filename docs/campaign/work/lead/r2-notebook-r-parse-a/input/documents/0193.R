compute_BF_distribution <- function(
  n,
  V_S_true,
  V_S_zero = 0.5,
  a = 1,
  b = 1,
  BF_alternative
) {
  # Values of the Bayes Factor
  BF_values <- numeric(n + 1)

  for (k in 0:n) {
    # Log beta-binomial marginal likelihood under the alternative hypothesis
    if (BF_alternative == "two_sided") {
      log_marginal_H1 <- lbeta(a + k, b + n - k) - lbeta(a, b)
    } else if (BF_alternative == "greater") {
      log_marginal_H1 <- lbeta(a + k, b + n - k) -
        lbeta(a, b) +
        log1p(-stats::pbeta(V_S_zero, a + k, b + n - k)) -
        log1p(-stats::pbeta(V_S_zero, a, b))
    }

    # Log binomial marginal likelihood under the null hypothesis
    log_likelihood_H0 <- k * log(V_S_zero) + (n - k) * log(1 - V_S_zero)

    # Bayes Factor
    BF_values[k + 1] <- exp(log_marginal_H1 - log_likelihood_H0)
  }

  # Probabilities of observing each Bayes Factor value
  BF_probs <- stats::dbinom(
    x = 0:n,
    size = n,
    prob = V_S_true
  )

  # Tabulate the distribution of the Bayes Factor
  # Note: Using .data$ to avoid 'no visible binding for global variable' notes in check
  BF_distribution <- data.frame(
    BF_values = BF_values,
    BF_probs = BF_probs
  ) %>%
    dplyr::group_by(.data$BF_values) %>%
    dplyr::summarise(BF_PMF = sum(.data$BF_probs), .groups = "drop") %>%
    dplyr::arrange(.data$BF_values) %>%
    dplyr::mutate(BF_CDF = cumsum(.data$BF_PMF))

  return(BF_distribution)
}