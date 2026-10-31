#' Winsorize a numeric vector
#'
#' @description
#' Replaces the lowest and highest `trim` fraction of the values by the
#' nearest retained value.
#'
#' @param x A numeric vector.
#' @param trim Fraction in `[0, 0.5)` to winsorize at each end.
#' @export
winsorize <- function(x, trim = 0.1) {
  if (trim < 0 || trim >= 0.5) {
    stop("`trim` must be in [0, 0.5).", call. = FALSE)
  }
  ok <- x[!is.na(x)]
  k <- floor(length(ok) * trim)
  if (k == 0L) {
    return(x)
  }

  sorted <- sort(ok)
  # Lower tail
  lower <- sorted[k + 1L]
  x[!is.na(x) & x < lower] <- lower
  # Upper tail
  upper <- sorted[length(sorted) - k]
  x[!is.na(x) & x > upper] <- upper
  x
}

#' Huber M-estimate of location
#'
#' @description
#' Iteratively reweighted mean with Huber weights, using the scaled median
#' absolute deviation as a fixed scale.
#'
#' @param x A numeric vector.
#' @param k Tuning constant; smaller values give more robustness.
#' @param tol Convergence tolerance on the location.
#' @param max.iter Maximum number of iterations.
#' @keywords internal
#' @return A single number with attribute `"iterations"`.
#' @noRd
huber.location <- function(x, k = 1.345, tol = 1e-8, max.iter = 50L) {
  x <- x[!is.na(x)]
  mu <- stats::median(x)
  s <- stats::mad(x)
  if (s == 0) {
    return(structure(mu, iterations = 0L))
  }

  # Reweight with the MAD scale held fixed until the location settles
  iter <- 0L
  repeat {
    iter <- iter + 1L
    r <- (x - mu) / s
    w <- pmin(1, k / abs(r))
    w[r == 0] <- 1
    mu.new <- sum(w * x) / sum(w)
    if (abs(mu.new - mu) < tol || iter >= max.iter) {
      break
    }
    mu <- mu.new
  }
  structure(mu.new, iterations = iter)
}

#' Hodges-Lehmann estimate of location
#'
#' @description The median of all pairwise Walsh averages.
#'
#' @param x A numeric vector.
#' @keywords internal
#' @noRd
hodges.lehmann <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(NA_real_)
  }
  walsh <- outer(x, x, "+") / 2
  stats::median(walsh[upper.tri(walsh, diag = TRUE)])
}

#' Estimate location with a chosen estimator
#'
#' @param x A numeric vector.
#' @param method One of `"mean"`, `"median"`, `"trimmed"`, `"winsorized"`,
#'   `"huber"` or `"hl"` (Hodges-Lehmann).
#' @param trim Fraction trimmed or winsorized at each end.
#' @export
loc.estimate <- function(x,
                         method = c("mean", "median", "trimmed", "winsorized", "huber", "hl"),
                         trim = 0.1) {
  method <- match.arg(method)
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  if (trim < 0 || trim >= 0.5) {
    stop("`trim` must be in [0, 0.5).", call. = FALSE)
  }
  x <- x[!is.na(x)]
  if (length(x) == 0L) {
    return(NA_real_)
  }

  switch(method,
    mean = {
      mean(x)
    },
    median = {
      stats::median(x)
    },
    trimmed = {
      mean(x, trim = trim)
    },
    winsorized = {
      mean(winsorize(x, trim = trim))
    },
    huber = {
      as.numeric(huber.location(x))
    },
    hl = {
      hodges.lehmann(x)
    }
  )
}

#' Standard error of a location estimate
#'
#' @description
#' Large-sample standard errors: the usual one for the mean, the kernel
#' density based one for the median, and the winsorized-variance based one
#' for the trimmed mean.
#'
#' @param x A numeric vector.
#' @param method One of `"mean"`, `"median"` or `"trimmed"`.
#' @param trim Fraction trimmed at each end for `"trimmed"`.
#' @keywords internal
loc.se <- function(x, method = c("mean", "median", "trimmed"), trim = 0.1) {
  method <- match.arg(method)
  x <- x[!is.na(x)]
  n <- length(x)

  switch(method,
    mean = {
      stats::sd(x) / sqrt(n)
    },
    median = {
      dens <- stats::density(x)
      f.med <- stats::approx(dens$x, dens$y, xout = stats::median(x))$y
      1 / (2 * f.med * sqrt(n))
    },
    trimmed = {
      sw <- stats::sd(winsorize(x, trim = trim))
      sw / ((1 - 2 * trim) * sqrt(n))
    }
  )
}

#' Tabulate location estimates for several columns
#'
#' @param data A data frame.
#' @param cols Character vector of numeric column names.
#' @param methods Character vector of methods accepted by [loc.estimate()].
#' @param trim Fraction trimmed or winsorized at each end.
#' @export
#' @return A matrix with one row per column and one column per method,
#'   with attribute `"n"` giving the non-missing count of every column.
loc.table <- function(data, cols = names(data), methods = c("mean", "median", "trimmed"),
                      trim = 0.1) {
  bad <- cols[!vapply(data[cols], is.numeric, logical(1))]
  if (length(bad) > 0L) {
    stop("Non-numeric columns: ", paste(bad, collapse = ", "), call. = FALSE)
  }
  out <- matrix(NA_real_, nrow = length(cols), ncol = length(methods),
                dimnames = list(cols, methods))
  n <- integer(length(cols))

  for (i in seq_along(cols)) {
    x <- data[[cols[i]]]
    n[i] <- sum(!is.na(x))
    for (m in methods) {
      out[i, m] <- loc.estimate(x, method = m, trim = trim)
    }
  }
  attr(out, "n") <- stats::setNames(n, cols)
  out
}

#' Summarise a sample with every location estimator
#'
#' @param x A numeric vector.
#' @param trim Fraction trimmed or winsorized at each end.
#' @export
#' @return A list with elements `n`, `mean`, `mean.se`, `median`,
#'   `median.se`, `trimmed`, `trimmed.se`, `winsorized`, `huber`,
#'   `huber.iter` and `hl`.
loc.summary <- function(x, trim = 0.1) {
  x <- x[!is.na(x)]
  n <- length(x)
  if (n < 3L) {
    stop("`x` must have at least 3 non-missing values.", call. = FALSE)
  }

  # Mean
  mean.est <- loc.estimate(x, "mean")
  mean.se <- loc.se(x, "mean")
  # Median
  median.est <- loc.estimate(x, "median")
  median.se <- loc.se(x, "median")
  # Trimmed mean
  trimmed.est <- loc.estimate(x, "trimmed", trim = trim)
  trimmed.se <- loc.se(x, "trimmed", trim = trim)
  # Winsorized mean
  winsorized.est <- loc.estimate(x, "winsorized", trim = trim)
  # Huber M-estimate
  huber <- huber.location(x)
  # Hodges-Lehmann
  hl.est <- hodges.lehmann(x)

  list(
    n = n,
    mean = mean.est,
    mean.se = mean.se,
    median = median.est,
    median.se = median.se,
    trimmed = trimmed.est,
    trimmed.se = trimmed.se,
    winsorized = winsorized.est,
    huber = as.numeric(huber),
    huber.iter = attr(huber, "iterations"),
    hl = hl.est
  )
}
