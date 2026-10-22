#' Kish design effect due to unequal weighting
#'
#' Computes `1 + cv^2` of the weights, the approximate variance inflation
#' caused by weighting alone.
#'
#' @param weights A numeric vector of respondent weights.
#' @return A single number, at least one.
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
#' @export
weight_cv <- function(weights) {
  weights <- weights[!is.na(weights)]
  stats::sd(weights) / mean(weights)
}

#' Trim extreme weights
#'
#' Caps weights at `cap` times their median and redistributes the trimmed mass
#' over the untrimmed weights, repeating until no weight exceeds the cap.
#'
#' @param weights A numeric vector of respondent weights.
#' @param cap Maximum allowed ratio of a weight to the median weight.
#' @param max_iter Maximum number of trim-and-redistribute passes.
#' @return A numeric vector of trimmed weights with the same total. Missing
#'   weights stay missing.
#' @export
trim_weights_cap <- function(weights, cap = 5, max_iter = 50L) {
  ok <- !is.na(weights)
  w <- weights[ok]
  total <- sum(w)
  limit <- cap * stats::median(w)

  # Redistributing the excess can push other weights over the limit, hence
  # the repeated passes.
  for (i in seq_len(max_iter)) {
    over <- w > limit
    if (!any(over) || all(over)) {
      break
    }
    excess <- sum(w[over] - limit)
    w[over] <- limit
    w[!over] <- w[!over] + excess * w[!over] / sum(w[!over])
  }

  weights[ok] <- w * total / sum(w)
  weights
}

#' @rdname trim_weights_cap
#' @description `trim_weights_quantile()` instead winsorises weights to the
#'   `lower` and `upper` quantiles and rescales so the weights keep their
#'   original total.
#' @param lower,upper Quantile probabilities bounding the weights.
#' @export
trim_weights_quantile <- function(weights, lower = 0.01, upper = 0.99) {
  if (lower >= upper) {
    stop("`lower` must be smaller than `upper`.", call. = FALSE)
  }

  ok <- !is.na(weights)
  w <- weights[ok]
  total <- sum(w)
  bounds <- stats::quantile(w, c(lower, upper), names = FALSE)
  w <- pmin(pmax(w, bounds[1]), bounds[2])

  weights[ok] <- w * total / sum(w)
  weights
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
  if (length(w) == 0) {
    stop("`weights` has no non-missing values.", call. = FALSE)
  }

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

  # Labels are padded by hand so the colons line up.
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
