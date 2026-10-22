#' @importFrom magrittr %>%
NULL

#' Convert a draws matrix to long format
#'
#' Turns an iterations-by-variables matrix, as returned by
#' `posterior::as_draws_matrix()` or `as.matrix()` on a fit, into a tibble.
#'
#' @param draws A numeric matrix of posterior draws with one column per
#'   variable.
#' @param chains Optional integer vector giving the chain of each row.
#' @return A tibble with columns `.draw`, `.chain`, `variable` and `value`.
#' @export
draws_long <- function(draws, chains = NULL) {
  if (!is.matrix(draws) || is.null(colnames(draws))) {
    stop("`draws` must be a matrix with column names.", call. = FALSE)
  }
  if (is.null(chains)) {
    chains <- rep(1L, nrow(draws))
  }
  draws %>%
    tibble::as_tibble() %>%
    dplyr::mutate(.draw = dplyr::row_number(), .chain = as.integer(chains)) %>%
    tidyr::pivot_longer(
      cols = -c(.draw, .chain),
      names_to = "variable",
      values_to = "value"
    )
}

#' Split-chain R-hat for one variable
#'
#' Splits each chain in half and compares between- and within-chain variance.
#'
#' @param value Numeric vector of draws for one variable.
#' @param chain Integer vector giving the chain of each draw.
#' @return A single number; values near 1 indicate convergence.
#' @keywords internal
split_rhat <- function(value, chain) {
  halves <- unlist(lapply(split(value, chain), function(v) {
    n <- length(v) %/% 2
    list(v[seq_len(n)], v[n + seq_len(n)])
  }), recursive = FALSE)
  n <- min(lengths(halves))
  if (n < 2) {
    return(NA_real_)
  }
  m <- vapply(halves, function(h) mean(h[seq_len(n)]), numeric(1))
  w <- mean(vapply(halves, function(h) stats::var(h[seq_len(n)]), numeric(1)))
  b <- n * stats::var(m)
  var_plus <- (n - 1) / n * w + b / n
  sqrt(var_plus / w)
}

#' Summarise posterior draws per variable
#'
#' Computes the posterior mean, standard deviation, central interval and
#' split R-hat for every variable.
#'
#' @param draws A numeric matrix of posterior draws with one column per
#'   variable.
#' @param chains Optional integer vector giving the chain of each row.
#' @param probs Lower and upper probabilities of the central interval.
#' @return A tibble with one row per variable.
#' @export
summarise_draws_simple <- function(draws, chains = NULL, probs = c(0.05, 0.95)) {
  if (length(probs) != 2L || any(probs <= 0 | probs >= 1) || probs[1] >= probs[2]) {
    stop("`probs` must be two increasing probabilities in (0, 1).", call. = FALSE)
  }
  draws_long(draws, chains) %>%
    dplyr::group_by(variable) %>%
    dplyr::summarise(
      mean = mean(value),
      sd = stats::sd(value),
      lower = stats::quantile(value, probs[1], names = FALSE),
      upper = stats::quantile(value, probs[2], names = FALSE),
      rhat = split_rhat(value, .chain),
      .groups = "drop"
    ) %>%
    dplyr::arrange(match(variable, colnames(draws)))
}

#' Relabel indexed Stan parameters with group names
#'
#' Replaces numeric indices such as `alpha[3]` with the group labels kept by
#' the `standata_*()` builders, e.g. `alpha[riverbank]`.
#'
#' @param summary A tibble with a `variable` column, as returned by
#'   [summarise_draws_simple()].
#' @param parameter Base name of the indexed parameter, e.g. `"alpha"`.
#' @param levels Character vector of group labels in index order.
#' @return `summary` with relabelled variables and an added `group` column.
#' @export
relabel_indexed <- function(summary, parameter, levels) {
  pattern <- paste0("^", parameter, "\\[(\\d+)\\]$")
  summary %>%
    dplyr::mutate(
      index = suppressWarnings(as.integer(sub(pattern, "\\1", variable))),
      index = dplyr::if_else(grepl(pattern, variable), index, NA_integer_),
      group = levels[index],
      variable = dplyr::if_else(
        is.na(index),
        variable,
        paste0(parameter, "[", group, "]")
      )
    ) %>%
    dplyr::select(-index)
}

#' Posterior probability that a quantity exceeds a threshold
#'
#' @param draws A numeric matrix of posterior draws with one column per
#'   variable.
#' @param variables Character vector of column names to evaluate.
#' @param threshold Value to compare against.
#' @return A named numeric vector of probabilities.
#' @export
prob_above <- function(draws, variables = colnames(draws), threshold = 0) {
  unknown <- setdiff(variables, colnames(draws))
  if (length(unknown) > 0L) {
    stop("Unknown variable(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  colMeans(draws[, variables, drop = FALSE] > threshold)
}

#' Posterior probability that a quantity falls below a threshold
#'
#' @inheritParams prob_above
#' @return A named numeric vector of probabilities.
#' @export
prob_below <- function(draws, variables = colnames(draws), threshold = 0) {
  unknown <- setdiff(variables, colnames(draws))
  if (length(unknown) > 0L) {
    stop("Unknown variable(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
  colMeans(draws[, variables, drop = FALSE] < threshold)
}

#' Posterior predictive counts from a Poisson model
#'
#' Simulates replicated counts for each observation from posterior draws of
#' the log rate.
#'
#' @param log_rate A draws-by-observations matrix of linear predictors on the
#'   log scale, including the offset.
#' @param n_draws Number of posterior draws to use, sampled without
#'   replacement.
#' @return An integer matrix with `n_draws` rows and one column per
#'   observation.
#' @export
predict_poisson_counts <- function(log_rate, n_draws = 200L) {
  if (!is.matrix(log_rate)) {
    stop("`log_rate` must be a matrix.", call. = FALSE)
  }
  n_draws <- min(n_draws, nrow(log_rate))
  rows <- sample.int(nrow(log_rate), n_draws)
  lambda <- exp(log_rate[rows, , drop = FALSE])
  y_rep <- matrix(
    stats::rpois(length(lambda), lambda),
    nrow = n_draws,
    dimnames = list(NULL, colnames(log_rate))
  )
  y_rep
}
