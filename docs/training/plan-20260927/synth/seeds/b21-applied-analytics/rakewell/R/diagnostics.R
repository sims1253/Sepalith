#' Kish design effect due to unequal weighting
#'
#' Computes `1 + cv^2` of the weights, the approximate variance inflation
#' caused by weighting alone.
#'
#' @param weights A numeric vector of respondent weights.
#' @return A single number, at least one.
#' @family weight diagnostics
#' @export
kish_deff <- function(weights) {
  weights <- weights[!is.na(weights)]
  n <- length(weights)
  n * sum(weights^2) / sum(weights)^2
}

#' Effective sample size under weighting
#'
#' Divides the nominal sample size by the Kish design effect.
#'
#' @param weights A numeric vector of respondent weights.
#' @return A single number between one and the number of weights.
#' @family weight diagnostics
#' @export
effective_n <- function(weights) {
  weights <- weights[!is.na(weights)]
  sum(weights)^2 / sum(weights^2)
}

#' Coefficient of variation of the weights
#'
#' Standard deviation of the weights divided by their mean.
#'
#' @param weights A numeric vector of respondent weights.
#' @return A single non-negative number.
#' @family weight diagnostics
#' @export
weight_cv <- function(weights) {
  weights <- weights[!is.na(weights)]
  stats::sd(weights) / mean(weights)
}

#' Trim weights at a fixed cap
#'
#' Caps weights at `cap` times their median and redistributes the trimmed mass
#' over the untrimmed weights, repeating until no weight exceeds the cap.
#'
#' @param weights A numeric vector of respondent weights.
#' @param cap Maximum allowed ratio of a weight to the median weight.
#' @param max_iter Maximum number of trim-and-redistribute passes.
#' @return A numeric vector of trimmed weights with the same total.
#' @family weight trimming
#' @export
trim_weights_cap <- function(weights, cap = 5, max_iter = 50L) {
  total <- sum(weights)
  limit <- cap * stats::median(weights)
  for (i in seq_len(max_iter)) {
    over <- weights > limit
    if (!any(over)) {
      break
    }
    excess <- sum(weights[over] - limit)
    weights[over] <- limit
    weights[!over] <- weights[!over] + excess * weights[!over] / sum(weights[!over])
  }
  weights * total / sum(weights)
}

#' Trim weights at upper and lower quantiles
#'
#' Winsorises weights to the `lower` and `upper` quantiles and rescales so the
#' weights keep their original total.
#'
#' @param weights A numeric vector of respondent weights.
#' @param lower,upper Quantile probabilities bounding the weights.
#' @return A numeric vector of trimmed weights with the same total.
#' @family weight trimming
#' @export
trim_weights_quantile <- function(weights, lower = 0.01, upper = 0.99) {
  if (lower >= upper) {
    stop("`lower` must be smaller than `upper`.", call. = FALSE)
  }
  total <- sum(weights)
  bounds <- stats::quantile(weights, c(lower, upper), names = FALSE)
  weights <- pmin(pmax(weights, bounds[1]), bounds[2])
  weights * total / sum(weights)
}

#' Summarise a weight vector
#'
#' Collects the common weighting diagnostics in one object of class
#' `weight_report`.
#'
#' @param weights A numeric vector of respondent weights.
#' @param label A short name for the weighting scheme, used when printing.
#' @return An object of class `weight_report`.
#' @export
weight_report <- function(weights, label = "weights") {
  if (!is.numeric(weights)) {
    stop("`weights` must be numeric.", call. = FALSE)
  }
  n_missing <- sum(is.na(weights))
  w <- weights[!is.na(weights)]
  structure(
    list(
      label = label,
      n = length(w),
      n_missing = n_missing,
      total = sum(w),
      range = range(w),
      ratio = max(w) / min(w),
      cv = weight_cv(w),
      deff = kish_deff(w),
      effective_n = effective_n(w)
    ),
    class = "weight_report"
  )
}

#' Format a weight report
#'
#' Turns a `weight_report` into a character vector of aligned lines.
#'
#' @param x A `weight_report` object.
#' @param digits Number of significant digits to show.
#' @param ... Unused, for compatibility with [format()].
#' @return A character vector, one element per line.
#' @export
format.weight_report <- function(x, digits = 3, ...) {
  fmt <- function(v) format(signif(v, digits), big.mark = ",")
  c(
    sprintf("<weight_report: %s>", x$label),
    sprintf("  n            : %s (%d missing)", fmt(x$n), x$n_missing),
    sprintf("  total        : %s", fmt(x$total)),
    sprintf("  range        : %s - %s", fmt(x$range[1]), fmt(x$range[2])),
    sprintf("  max/min      : %s", fmt(x$ratio)),
    sprintf("  cv           : %s", fmt(x$cv)),
    sprintf("  design effect: %s", fmt(x$deff)),
    sprintf("  effective n  : %s", fmt(x$effective_n))
  )
}

#' Print a weight report
#'
#' Writes the formatted report to the console, one diagnostic per line.
#'
#' @param x A `weight_report` object.
#' @param ... Passed to [format.weight_report()].
#' @return `x`, invisibly.
#' @export
print.weight_report <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
  invisible(x)
}
