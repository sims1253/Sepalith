#' Control chart constants
#'
#' Returns the bias-correction constants for subgroup ranges and standard
#' deviations. `d2` and `d3` come from the usual tables (n = 2 to 10);
#' `c4` is computed exactly.
#'
#' @param n Subgroup size.
#' @return A named list with elements `d2`, `d3` and `c4`.
#' @keywords internal
qc.constants <- function(n) {
  if (length(n) != 1L || n < 2 || n != round(n)) {
    stop("Subgroup size must be a single integer >= 2.", call. = FALSE)
  }

  # Tabulated for n = 2..10; larger subgroups should use the S chart.
  d2.table <- c(1.128, 1.693, 2.059, 2.326, 2.534, 2.704, 2.847, 2.970, 3.078)
  d3.table <- c(0.853, 0.888, 0.880, 0.864, 0.848, 0.833, 0.820, 0.808, 0.797)
  if (n > 10) {
    d2 <- NA_real_
    d3 <- NA_real_
  } else {
    d2 <- d2.table[n - 1L]
    d3 <- d3.table[n - 1L]
  }

  c4 <- sqrt(2 / (n - 1)) * exp(lgamma(n / 2) - lgamma((n - 1) / 2))
  list(d2 = d2, d3 = d3, c4 = c4)
}

#' Assemble a control limits object
#'
#' Shared constructor for every Shewhart chart type in this file.
#'
#' @param stat Numeric vector of plotted statistics, one per subgroup.
#' @param center Center line.
#' @param lcl,ucl Lower and upper control limits, scalars or vectors.
#' @param type Chart type label.
#' @param sigma Estimated process standard deviation, if relevant.
#' @param k Width of the limits in standard errors, kept so that the run
#'   rules can recover the sigma zones.
#' @return An object of class `qc_shewhart`.
#' @noRd
new.qc.shewhart <- function(stat, center, lcl, ucl, type, sigma = NA_real_, k = 3) {
  lcl <- rep_len(lcl, length(stat))
  ucl <- rep_len(ucl, length(stat))
  # which() drops NA, so missing subgroups never count as signals.
  out.of.control <- stat < lcl | stat > ucl

  structure(
    list(
      stat = stat,
      center = center,
      lcl = lcl,
      ucl = ucl,
      sigma = sigma,
      k = k,
      type = type,
      signals = which(out.of.control)
    ),
    class = "qc_shewhart"
  )
}

#' Variables control charts
#'
#' Limits for subgrouped measurements. `xbar.limits()` charts subgroup means,
#' estimating sigma from the average range unless it is supplied;
#' `r.limits()` charts subgroup ranges using the `d2` and `d3` constants;
#' `s.limits()` charts subgroup standard deviations using `c4`. Incomplete
#' subgroups get a missing range or standard deviation, and missing
#' statistics are left out of the center line.
#'
#' @param x A numeric matrix with one subgroup per row.
#' @param sigma Optional known process standard deviation.
#' @param k Width of the limits in standard errors.
#' @return An object of class `qc_shewhart`.
#' @export
xbar.limits <- function(x, sigma = NULL, k = 3) {
  x <- as.matrix(x)
  n <- ncol(x)
  if (n < 2L) {
    stop("`x` needs at least two observations per subgroup.", call. = FALSE)
  }
  stat <- rowMeans(x, na.rm = TRUE)
  stat[is.nan(stat)] <- NA_real_
  center <- mean(stat, na.rm = TRUE)

  if (is.null(sigma)) {
    complete <- stats::complete.cases(x)
    ranges <- apply(x[complete, , drop = FALSE], 1L, function(row) diff(range(row)))
    sigma <- mean(ranges) / qc.constants(n)$d2
  }
  se <- sigma / sqrt(n)
  new.qc.shewhart(stat, center, center - k * se, center + k * se, "xbar", sigma, k)
}

#' @rdname xbar.limits
#' @export
r.limits <- function(x, k = 3) {
  x <- as.matrix(x)
  n <- ncol(x)
  if (n < 2L) {
    stop("`x` needs at least two observations per subgroup.", call. = FALSE)
  }
  # Ranges are only comparable between complete subgroups.
  stat <- apply(x, 1L, function(row) {
    if (anyNA(row)) NA_real_ else diff(range(row))
  })
  center <- mean(stat, na.rm = TRUE)

  const <- qc.constants(n)
  sigma <- center / const$d2
  spread <- k * const$d3 * sigma
  new.qc.shewhart(stat, center, max(0, center - spread), center + spread, "r", sigma, k)
}

#' @rdname xbar.limits
#' @export
s.limits <- function(x, k = 3) {
  x <- as.matrix(x)
  n <- ncol(x)
  if (n < 2L) {
    stop("`x` needs at least two observations per subgroup.", call. = FALSE)
  }
  stat <- apply(x, 1L, function(row) {
    if (anyNA(row)) NA_real_ else stats::sd(row)
  })
  center <- mean(stat, na.rm = TRUE)

  c4 <- qc.constants(n)$c4
  sigma <- center / c4
  spread <- k * sigma * sqrt(1 - c4^2)
  new.qc.shewhart(stat, center, max(0, center - spread), center + spread, "s", sigma, k)
}

#' P chart limits
#'
#' Computes nonconforming proportions with limits that vary with the
#' inspected sample size.
#'
#' @param defects Integer vector of nonconforming counts per sample.
#' @param sizes Integer vector of sample sizes.
#' @param k Width of the limits in standard errors.
#' @return An object of class `qc_shewhart`.
#' @export
p.limits <- function(defects, sizes, k = 3) {
  if (length(defects) != length(sizes)) {
    stop("`defects` and `sizes` must have the same length.", call. = FALSE)
  }
  if (any(defects > sizes)) {
    stop("A sample cannot have more defects than items.", call. = FALSE)
  }

  stat <- defects / sizes
  center <- sum(defects) / sum(sizes)
  # Binomial standard error, so the limits widen for small samples.
  se <- sqrt(center * (1 - center) / sizes)
  lcl <- pmax(0, center - k * se)
  ucl <- pmin(1, center + k * se)
  new.qc.shewhart(stat, center, lcl, ucl, "p", k = k)
}

#' C chart limits
#'
#' Computes limits for counts of defects per inspection unit under a Poisson
#' model.
#'
#' @param counts Integer vector of defect counts per unit.
#' @param k Width of the limits in standard errors.
#' @return An object of class `qc_shewhart`.
#' @export
count.limits <- function(counts, k = 3) {
  if (any(counts < 0)) {
    stop("`counts` must be non-negative.", call. = FALSE)
  }

  stat <- as.numeric(counts)
  center <- mean(stat)
  # Poisson counts: the variance equals the mean.
  se <- sqrt(center)
  lcl <- max(0, center - k * se)
  ucl <- center + k * se
  new.qc.shewhart(stat, center, lcl, ucl, "c", k = k)
}

#' Western Electric run rules
#'
#' Flags points that violate the four classic run rules relative to the center
#' line and the one-, two- and three-sigma zones, plus a six-point trend rule.
#' Sigma zones are derived from the upper limit and the chart's `k`. Missing
#' points never trigger a rule.
#'
#' @param chart An object of class `qc_shewhart`.
#' @return A data frame with one logical column per rule and one row per point.
#' @export
we.rules <- function(chart) {
  stopifnot(inherits(chart, "qc_shewhart"))
  k <- if (is.null(chart$k)) 3 else chart$k
  z <- (chart$stat - chart$center) / ((chart$ucl - chart$center) / k)
  n <- length(z)

  rule1 <- !is.na(z) & abs(z) > 3
  rule2 <- logical(n)
  rule3 <- logical(n)
  rule4 <- logical(n)
  rule5 <- logical(n)
  for (i in seq_len(n)) {
    if (i >= 3L) {
      w <- z[(i - 2L):i]
      rule2[i] <- sum(w > 2, na.rm = TRUE) >= 2L || sum(w < -2, na.rm = TRUE) >= 2L
    }
    if (i >= 5L) {
      w <- z[(i - 4L):i]
      rule3[i] <- sum(w > 1, na.rm = TRUE) >= 4L || sum(w < -1, na.rm = TRUE) >= 4L
    }
    if (i >= 8L) {
      w <- z[(i - 7L):i]
      rule4[i] <- isTRUE(all(w > 0)) || isTRUE(all(w < 0))
    }
    if (i >= 6L) {
      steps <- diff(z[(i - 5L):i])
      rule5[i] <- isTRUE(all(steps > 0)) || isTRUE(all(steps < 0))
    }
  }

  data.frame(
    index = seq_len(n),
    beyond.3sigma = rule1,
    two.of.three = rule2,
    four.of.five = rule3,
    eight.one.side = rule4,
    six.trending = rule5,
    any = rule1 | rule2 | rule3 | rule4 | rule5
  )
}
