#' Tabular CUSUM
#'
#' Accumulates standardized deviations above and below the target, each
#' reduced by the reference value `k`, and signals when either sum exceeds
#' the decision interval `h`.
#'
#' @param x Numeric vector of individual observations.
#' @param target In-control process mean.
#' @param sigma Process standard deviation; estimated from moving ranges when
#'   `NULL`.
#' @param k Reference value in standard deviations.
#' @param h Decision interval in standard deviations.
#' @param fir Fast initial response head start, as a fraction of `h`.
#' @return An object of class `qc_cusum`.
#' @export
cusum.tabular <- function(x, target, sigma = NULL, k = 0.5, h = 5, fir = 0) {
  if (!is.numeric(x) || length(x) < 2L) {
    stop("`x` must be a numeric vector with at least two values.", call. = FALSE)
  }
  if (fir < 0 || fir >= 1) {
    stop("`fir` must be in [0, 1).", call. = FALSE)
  }

  # Moving-range estimate: mean |x_t - x_{t-1}| / d2 with d2 = 1.128.
  if (is.null(sigma)) {
    sigma <- mean(abs(diff(x)), na.rm = TRUE) / 1.128
  }
  z <- (x - target) / sigma

  n <- length(z)
  upper <- numeric(n)
  lower <- numeric(n)
  prev.upper <- fir * h
  prev.lower <- fir * h
  for (i in seq_len(n)) {
    # A missing observation leaves both sums where they were.
    if (is.na(z[i])) {
      upper[i] <- prev.upper
      lower[i] <- prev.lower
    } else {
      upper[i] <- max(0, prev.upper + z[i] - k)
      lower[i] <- max(0, prev.lower - z[i] - k)
    }
    prev.upper <- upper[i]
    prev.lower <- lower[i]
  }

  return(structure(
    list(
      x = x,
      upper = upper,
      lower = lower,
      target = target,
      sigma = sigma,
      k = k,
      h = h,
      signals.upper = which(upper > h),
      signals.lower = which(lower > h)
    ),
    class = "qc_cusum"
  ))
}

#' CUSUM with restart after each signal
#'
#' Like [cusum.tabular()] but resets both sums to zero after an alarm, which
#' mimics an investigated and corrected process. Missing observations are
#' skipped in the same way.
#'
#' @inheritParams cusum.tabular
#' @return An object of class `qc_cusum`.
#' @export
cusum.restart <- function(x, target, sigma = NULL, k = 0.5, h = 5) {
  if (!is.numeric(x) || length(x) < 2L) {
    stop("`x` must be a numeric vector with at least two values.", call. = FALSE)
  }

  if (is.null(sigma)) {
    sigma <- mean(abs(diff(x)), na.rm = TRUE) / 1.128
  }
  z <- (x - target) / sigma

  n <- length(z)
  upper <- numeric(n)
  lower <- numeric(n)
  prev.upper <- 0
  prev.lower <- 0
  for (i in seq_len(n)) {
    if (is.na(z[i])) {
      upper[i] <- prev.upper
      lower[i] <- prev.lower
      next
    }
    upper[i] <- max(0, prev.upper + z[i] - k)
    lower[i] <- max(0, prev.lower - z[i] - k)
    if (upper[i] > h || lower[i] > h) {
      prev.upper <- 0
      prev.lower <- 0
    } else {
      prev.upper <- upper[i]
      prev.lower <- lower[i]
    }
  }

  return(structure(
    list(
      x = x,
      upper = upper,
      lower = lower,
      target = target,
      sigma = sigma,
      k = k,
      h = h,
      signals.upper = which(upper > h),
      signals.lower = which(lower > h)
    ),
    class = "qc_cusum"
  ))
}

#' Locate CUSUM signals and estimated change points
#'
#' For every first alarm in a run of alarms, walks back to the last zero of
#' the sum to estimate when the shift started and estimates its size.
#'
#' @param object An object of class `qc_cusum`.
#' @return A data frame with one row per alarm run.
#' @export
cusum.signals <- function(object) {
  stopifnot(inherits(object, "qc_cusum"))

  rows <- list()
  for (side in c("upper", "lower")) {
    s <- object[[side]]
    idx <- object[[paste0("signals.", side)]]
    if (length(idx) == 0L) {
      next
    }
    # Only the first alarm of each consecutive run gets a row.
    first <- idx[c(TRUE, diff(idx) > 1L)]
    for (i in first) {
      # The shift is taken to start just after the sum last left zero.
      zeros <- which(s[seq_len(i)] == 0)
      start <- if (length(zeros) > 0L) max(zeros) + 1L else 1L
      run.length <- i - start + 1L
      direction <- if (side == "upper") 1 else -1
      shift <- direction * object$sigma * (object$k + s[i] / run.length)
      rows[[length(rows) + 1L]] <- data.frame(
        side = side,
        alarm = i,
        change.point = start,
        shift = shift,
        stringsAsFactors = FALSE
      )
    }
  }

  if (length(rows) == 0L) {
    return(data.frame(
      side = character(),
      alarm = integer(),
      change.point = integer(),
      shift = numeric()
    ))
  }
  out <- do.call(rbind, rows)
  return(out[order(out$alarm), , drop = FALSE])
}

#' Approximate average run length of a CUSUM
#'
#' Uses Siegmund's approximation for each one-sided scheme and combines them
#' into the two-sided run length.
#'
#' @param k Reference value in standard deviations.
#' @param h Decision interval in standard deviations.
#' @param delta Mean shift in standard deviations; 0 gives the in-control ARL.
#' @return A single numeric value.
#' @export
cusum.arl <- function(k = 0.5, h = 5, delta = 0) {
  # Siegmund's correction adds 1.166 to the decision interval.
  b <- h + 1.166
  one.sided <- function(d) {
    if (abs(d) < 1e-8) {
      return(b^2)
    }
    (exp(-2 * d * b) + 2 * d * b - 1) / (2 * d^2)
  }

  arl.upper <- one.sided(delta - k)
  arl.lower <- one.sided(-delta - k)
  # Two one-sided schemes combine like parallel hazards.
  return(1 / (1 / arl.upper + 1 / arl.lower))
}

#' Choose a decision interval for a target ARL
#'
#' Solves for the `h` that gives the requested in-control average run length
#' at a given reference value.
#'
#' @param target.arl Desired in-control average run length.
#' @param k Reference value in standard deviations.
#' @param interval Search interval for `h`.
#' @return A list with the chosen `h` and the run lengths it implies.
#' @export
cusum.design <- function(target.arl = 370, k = 0.5, interval = c(0.5, 20)) {
  if (target.arl <= 1) {
    stop("`target.arl` must be greater than 1.", call. = FALSE)
  }

  # Work on the log scale: the ARL grows roughly exponentially in h.
  objective <- function(h) log(cusum.arl(k = k, h = h)) - log(target.arl)
  root <- tryCatch(
    stats::uniroot(objective, interval = interval)$root,
    error = function(e) {
      stop("No decision interval found in `interval`; widen it.", call. = FALSE)
    }
  )

  return(list(
    h = root,
    k = k,
    arl.in.control = cusum.arl(k = k, h = root),
    arl.one.sigma = cusum.arl(k = k, h = root, delta = 1)
  ))
}

#' Print a CUSUM
#'
#' Shows the design parameters and the number of alarms on each side.
#'
#' @param x An object of class `qc_cusum`.
#' @param ... Unused.
#' @return `x`, invisibly.
#' @export
print.qc_cusum <- function(x, ...) {
  cat("<tabular CUSUM>\n")
  cat(sprintf("  target = %g, sigma = %g, k = %g, h = %g\n", x$target, x$sigma, x$k, x$h))
  cat(sprintf("  observations: %d\n", length(x$x)))
  cat(sprintf("  upper alarms: %d\n", length(x$signals.upper)))
  cat(sprintf("  lower alarms: %d\n", length(x$signals.lower)))
  return(invisible(x))
}

#' Summarise a CUSUM
#'
#' Collects the design, the alarm table and the largest excursion on each
#' side into a list.
#'
#' @param object An object of class `qc_cusum`.
#' @inheritParams print.qc_cusum
#' @return An object of class `summary.qc_cusum`.
#' @export
summary.qc_cusum <- function(object, ...) {
  out <- list(
    design = c(target = object$target, sigma = object$sigma, k = object$k, h = object$h),
    signals = cusum.signals(object),
    max.upper = max(object$upper),
    max.lower = max(object$lower),
    in.control = length(object$signals.upper) + length(object$signals.lower) == 0L
  )
  class(out) <- "summary.qc_cusum"
  return(out)
}
