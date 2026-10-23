#' @title Most frequent value
#'
#' @param x An atomic vector.
#' @param ties Which value to return when several share the top count:
#'   `"first"` (in sorted order) or `"all"`.
#' @keywords internal
mode_value <- function(x, ties = c("first", "all")) {
  ties <- match.arg(ties)
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(x)
  }
  values <- sort(unique(x))
  counts <- tabulate(match(x, values), nbins = length(values))
  top <- values[counts == max(counts)]
  if (ties == "first") top[1L] else top
}

#' @title Frequency table with cumulative columns
#'
#' @param x An atomic vector.
#' @param sort Logical; order rows by decreasing frequency instead of value?
#' @return A data frame with columns `value`, `n`, `prop`, `cum_n` and
#'   `cum_prop`.
#' @export
freq_table <- function(x, sort = TRUE) {
  x <- x[!is.na(x)]
  counts <- table(x)

  out <- data.frame(
    value = names(counts),
    n = as.integer(counts),
    stringsAsFactors = FALSE
  )
  if (sort) {
    out <- out[order(-out$n, out$value), , drop = FALSE]
  }
  out$prop <- out$n / sum(out$n)
  out$cum_n <- cumsum(out$n)
  out$cum_prop <- out$cum_n / sum(out$n)
  rownames(out) <- NULL
  out
}

#' @title Two-way table of counts or proportions
#'
#' @param x,y Atomic vectors of the same length giving the row and column
#'   classification.
#' @param margin What to report: `"count"` for raw counts, `"total"` for
#'   proportions of the grand total, `"row"` for row proportions or
#'   `"column"` for column proportions.
#' @param digits Number of decimal places for proportions, or `NULL`.
#' @export
prop_table_by <- function(x, y, margin = c("count", "total", "row", "column"),
                          digits = NULL) {
  margin <- match.arg(margin)
  if (length(x) != length(y)) {
    stop("`x` and `y` must have the same length.", call. = FALSE)
  }
  counts <- unclass(table(x, y))

  out <- switch(margin,
    count = {
      counts
    },
    total = {
      counts / sum(counts)
    },
    row = {
      totals <- rowSums(counts)
      sweep(counts, 1L, ifelse(totals > 0, totals, 1), "/")
    },
    column = {
      totals <- colSums(counts)
      sweep(counts, 2L, ifelse(totals > 0, totals, 1), "/")
    }
  )

  # Counts are left unrounded
  if (!is.null(digits) && margin != "count") {
    out <- round(out, digits)
  }
  out
}

#' @title Association statistics for a two-way table
#'
#' @param x,y Atomic vectors of the same length giving the row and column
#'   classification.
#' @return A list with elements `chisq`, `df`, `p_value`, `phi`, `cramer_v`
#'   and `contingency`.
#' @export
crosstab_stats <- function(x, y) {
  observed <- unclass(table(x, y))
  n <- sum(observed)

  # Expected counts under independence of the margins
  expected <- outer(rowSums(observed), colSums(observed)) / n
  chisq <- sum((observed - expected)^2 / expected)
  df <- (nrow(observed) - 1L) * (ncol(observed) - 1L)
  k <- min(dim(observed))

  list(
    chisq = chisq,
    df = df,
    p_value = stats::pchisq(chisq, df, lower.tail = FALSE),
    phi = sqrt(chisq / n),
    cramer_v = sqrt(chisq / (n * (k - 1L))),
    contingency = sqrt(chisq / (chisq + n))
  )
}

#' @title Risk measures from a binary exposure and outcome
#'
#' Confidence limits for the ratios use the usual log-scale standard errors.
#'
#' @param exposure Logical vector; `TRUE` for exposed units.
#' @param outcome Logical vector; `TRUE` for units with the event.
#' @param conf_level Confidence level for the ratio intervals.
#' @return A list with elements `risk_exposed`, `risk_unexposed`,
#'   `risk_ratio`, `rr_lower`, `rr_upper`, `risk_diff`, `odds_ratio`,
#'   `or_lower` and `or_upper`.
#' @export
risk_table <- function(exposure, outcome, conf_level = 0.95) {
  keep <- !is.na(exposure) & !is.na(outcome)
  exposure <- as.logical(exposure[keep])
  outcome <- as.logical(outcome[keep])
  z <- stats::qnorm(1 - (1 - conf_level) / 2)

  # Exposed group
  events_exposed <- sum(outcome[exposure])
  n_exposed <- sum(exposure)
  risk_exposed <- events_exposed / n_exposed
  odds_exposed <- events_exposed / (n_exposed - events_exposed)

  # Unexposed group
  events_unexposed <- sum(outcome[!exposure])
  n_unexposed <- sum(!exposure)
  risk_unexposed <- events_unexposed / n_unexposed
  odds_unexposed <- events_unexposed / (n_unexposed - events_unexposed)

  # Risk ratio on the log scale
  risk_ratio <- risk_exposed / risk_unexposed
  se_log_rr <- sqrt(1 / events_exposed - 1 / n_exposed +
                      1 / events_unexposed - 1 / n_unexposed)
  # Odds ratio on the log scale
  odds_ratio <- odds_exposed / odds_unexposed
  se_log_or <- sqrt(1 / events_exposed + 1 / (n_exposed - events_exposed) +
                      1 / events_unexposed + 1 / (n_unexposed - events_unexposed))

  list(
    risk_exposed = risk_exposed,
    risk_unexposed = risk_unexposed,
    risk_ratio = risk_ratio,
    rr_lower = exp(log(risk_ratio) - z * se_log_rr),
    rr_upper = exp(log(risk_ratio) + z * se_log_rr),
    risk_diff = risk_exposed - risk_unexposed,
    odds_ratio = odds_ratio,
    or_lower = exp(log(odds_ratio) - z * se_log_or),
    or_upper = exp(log(odds_ratio) + z * se_log_or)
  )
}

#' @title Summaries of a response within bins of a predictor
#'
#' @param x A numeric predictor used for binning.
#' @param y A numeric response, the same length as `x`.
#' @param breaks Either a single number of equal-width bins or a vector of
#'   break points passed to [cut()].
#' @return A data frame with columns `bin`, `n`, `x_mean`, `y_mean`,
#'   `y_sd`, `y_se`, `y_min`, `y_median` and `y_max`, one row per non-empty
#'   bin.
#' @export
binned_summary <- function(x, y, breaks = 5L) {
  if (length(x) != length(y)) {
    stop("`x` and `y` must have the same length.", call. = FALSE)
  }
  keep <- !is.na(x) & !is.na(y)
  x <- x[keep]
  y <- y[keep]
  bins <- cut(x, breaks = breaks, include.lowest = TRUE)
  x_parts <- split(x, bins, drop = TRUE)
  y_parts <- split(y, bins, drop = TRUE)

  n <- vapply(y_parts, length, integer(1))
  x_mean <- vapply(x_parts, mean, numeric(1))
  y_mean <- vapply(y_parts, mean, numeric(1))
  y_sd <- vapply(y_parts, function(v) {
    if (length(v) > 1L) stats::sd(v) else NA_real_
  }, numeric(1))
  y_se <- y_sd / sqrt(n)
  y_min <- vapply(y_parts, min, numeric(1))
  y_median <- vapply(y_parts, stats::median, numeric(1))
  y_max <- vapply(y_parts, max, numeric(1))

  data.frame(
    bin = names(y_parts),
    n = unname(n),
    x_mean = unname(x_mean),
    y_mean = unname(y_mean),
    y_sd = unname(y_sd),
    y_se = unname(y_se),
    y_min = unname(y_min),
    y_median = unname(y_median),
    y_max = unname(y_max),
    stringsAsFactors = FALSE
  )
}
