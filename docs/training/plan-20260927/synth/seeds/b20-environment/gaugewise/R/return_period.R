#' Assign dates to water years
#'
#' Maps calendar dates to the water year in which they fall, labelled by the
#' calendar year in which the water year ends.
#'
#' @param date A vector of class `Date`.
#' @param start_month Integer month (1-12) on which the water year starts.
#' @return An integer vector of water years the same length as `date`.
#' @export
water_year = function(date, start_month = 10L) {
  if (!inherits(date, "Date")) {
    stop("`date` must be a Date vector.", call. = FALSE)
  }
  yr = as.integer(format(date, "%Y"))
  mo = as.integer(format(date, "%m"))
  if (start_month == 1L) {
    return(yr)
  }
  ifelse(mo >= start_month, yr + 1L, yr)
}

#' Extract annual maximum series
#'
#' Finds the peak discharge in each water year, skipping years with too much
#' missing data.
#'
#' @param date A vector of class `Date`.
#' @param flow A numeric vector of discharge values matching `date`.
#' @param start_month Integer month (1-12) on which the water year starts.
#' @param max_missing Maximum fraction of missing days tolerated per year.
#' @return A data frame with columns `water_year`, `peak_date` and `peak_flow`.
#' @export
annual_maxima = function(date, flow, start_month = 10L, max_missing = 0.1) {
  wy = water_year(date, start_month = start_month)
  years = sort(unique(wy))
  out = data.frame(
    water_year = years,
    peak_date = as.Date(rep(NA_character_, length(years))),
    peak_flow = rep(NA_real_, length(years))
  )
  for (i in seq_along(years)) {
    in_year = which(wy == years[i])
    if (mean(is.na(flow[in_year])) > max_missing) {
      next
    }
    j = in_year[which.max(flow[in_year])]
    out$peak_date[i] = date[j]
    out$peak_flow[i] = flow[j]
  }
  out[!is.na(out$peak_flow), , drop = FALSE]
}

#' Fit a Gumbel distribution to annual maxima
#'
#' Estimates location and scale parameters by the method of moments.
#'
#' @param peaks A numeric vector of annual maximum discharges.
#' @return An object of class `gw_fit` with elements `dist`, `params` and `n`.
#' @family frequency fits
#' @export
fit_gumbel = function(peaks) {
  peaks = peaks[!is.na(peaks)]
  if (length(peaks) < 10) {
    warning("Fewer than 10 annual maxima; estimates will be unstable.", call. = FALSE)
  }
  scale = stats::sd(peaks) * sqrt(6) / pi
  location = mean(peaks) - 0.5772157 * scale
  structure(
    list(dist = "gumbel", params = c(location = location, scale = scale), n = length(peaks)),
    class = "gw_fit"
  )
}

#' Fit a log-Pearson type III distribution to annual maxima
#'
#' Estimates the mean, standard deviation and skew of base-10 logarithms of
#' the annual maxima.
#'
#' @param peaks A numeric vector of annual maximum discharges.
#' @return An object of class `gw_fit` with elements `dist`, `params` and `n`.
#' @family frequency fits
#' @export
fit_lp3 = function(peaks) {
  peaks = peaks[!is.na(peaks)]
  if (length(peaks) < 10) {
    warning("Fewer than 10 annual maxima; estimates will be unstable.", call. = FALSE)
  }
  if (any(peaks <= 0)) {
    stop("Log-Pearson III requires strictly positive peaks.", call. = FALSE)
  }
  lx = log10(peaks)
  n = length(lx)
  m = mean(lx)
  s = stats::sd(lx)
  g = n * sum((lx - m)^3) / ((n - 1) * (n - 2) * s^3)
  structure(
    list(dist = "lp3", params = c(mean = m, sd = s, skew = g), n = n),
    class = "gw_fit"
  )
}

#' Fit a generalised extreme value distribution to annual maxima
#'
#' Estimates location, scale and shape parameters from sample L-moments
#' using a rational approximation for the shape.
#'
#' @param peaks A numeric vector of annual maximum discharges.
#' @return An object of class `gw_fit` with elements `dist`, `params` and `n`.
#' @family frequency fits
#' @export
fit_gev = function(peaks) {
  peaks = peaks[!is.na(peaks)]
  if (length(peaks) < 10) {
    warning("Fewer than 10 annual maxima; estimates will be unstable.", call. = FALSE)
  }
  x = sort(peaks)
  n = length(x)
  i = seq_len(n)
  b0 = mean(x)
  b1 = sum((i - 1) / (n - 1) * x) / n
  b2 = sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
  l1 = b0
  l2 = 2 * b1 - b0
  t3 = (6 * b2 - 6 * b1 + b0) / l2
  cc = 2 / (3 + t3) - log(2) / log(3)
  shape = 7.8590 * cc + 2.9554 * cc^2
  scale = l2 * shape / ((1 - 2^(-shape)) * gamma(1 + shape))
  location = l1 + scale * (gamma(1 + shape) - 1) / shape
  structure(
    list(
      dist = "gev",
      params = c(location = location, scale = scale, shape = shape),
      n = n
    ),
    class = "gw_fit"
  )
}

#' Design discharge for given return periods
#'
#' Evaluates the fitted distribution's quantile function at non-exceedance
#' probabilities `1 - 1 / return_period`.
#'
#' @param fit An object of class `gw_fit`.
#' @param return_period Numeric vector of return periods in years, all
#'   greater than 1.
#' @return A data frame with columns `return_period`, `probability` and
#'   `discharge`.
#' @export
design_flow = function(fit, return_period = c(2, 5, 10, 25, 50, 100)) {
  if (!inherits(fit, "gw_fit")) {
    stop("`fit` must be a `gw_fit` object.", call. = FALSE)
  }
  if (any(return_period <= 1)) {
    stop("`return_period` values must exceed 1.", call. = FALSE)
  }
  p = 1 - 1 / return_period
  par = fit$params
  q = switch(fit$dist,
    gumbel = par[["location"]] - par[["scale"]] * log(-log(p)),
    lp3 = {
      z = stats::qnorm(p)
      g = par[["skew"]]
      k = if (abs(g) < 1e-6) z else (2 / g) * ((1 + g * z / 6 - g^2 / 36)^3 - 1)
      10^(par[["mean"]] + k * par[["sd"]])
    },
    gev = par[["location"]] +
      par[["scale"]] / par[["shape"]] * (1 - (-log(p))^par[["shape"]]),
    stop("Unknown distribution: ", fit$dist, call. = FALSE)
  )
  data.frame(return_period = return_period, probability = p, discharge = q)
}

#' Print a frequency fit
#'
#' @param x An object of class `gw_fit`.
#' @param digits Number of significant digits for parameter values.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
print.gw_fit = function(x, digits = 4, ...) {
  cat("<gw_fit> ", x$dist, " fitted to ", x$n, " annual maxima\n", sep = "")
  for (nm in names(x$params)) {
    cat("  ", formatC(nm, width = -9), signif(x$params[[nm]], digits), "\n", sep = "")
  }
  invisible(x)
}

#' Bootstrap confidence intervals for design floods
#'
#' Resamples the annual maxima with replacement, refits the chosen
#' distribution and collects design discharges to form percentile intervals.
#'
#' @param peaks A numeric vector of annual maximum discharges.
#' @param dist Distribution to fit, one of `"gumbel"`, `"lp3"` or `"gev"`.
#' @param return_period Numeric vector of return periods in years, all
#'   greater than 1.
#' @param n_boot Number of bootstrap resamples.
#' @param level Confidence level of the interval.
#' @param seed Optional random seed for reproducibility.
#' @return A data frame with columns `return_period`, `estimate`, `lower`,
#'   `upper` and `n_failed`.
#' @export
design_flow_ci = function(peaks, dist = c("gumbel", "lp3", "gev"),
                          return_period = c(10, 50, 100), n_boot = 1000,
                          level = 0.9, seed = NULL) {
  dist = match.arg(dist)
  peaks = peaks[!is.na(peaks)]
  fitter = switch(dist,
    gumbel = fit_gumbel,
    lp3 = fit_lp3,
    gev = fit_gev
  )
  if (!is.null(seed)) {
    set.seed(seed)
  }
  estimate = design_flow(fitter(peaks), return_period)$discharge
  boot = matrix(NA_real_, nrow = n_boot, ncol = length(return_period))
  n_failed = 0L
  for (b in seq_len(n_boot)) {
    sample_b = sample(peaks, replace = TRUE)
    q = tryCatch(
      suppressWarnings(design_flow(fitter(sample_b), return_period)$discharge),
      error = function(e) NULL
    )
    if (is.null(q) || any(!is.finite(q))) {
      n_failed = n_failed + 1L
      next
    }
    boot[b, ] = q
  }
  alpha = (1 - level) / 2
  lower = apply(boot, 2, stats::quantile, probs = alpha, na.rm = TRUE, names = FALSE)
  upper = apply(boot, 2, stats::quantile, probs = 1 - alpha, na.rm = TRUE, names = FALSE)
  if (n_failed > 0.1 * n_boot) {
    warning(sprintf("%d of %d resamples failed to fit.", n_failed, n_boot), call. = FALSE)
  }
  data.frame(
    return_period = return_period,
    estimate = estimate,
    lower = lower,
    upper = upper,
    n_failed = n_failed
  )
}
