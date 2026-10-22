#' Check daily temperature inputs
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @noRd
checkTemps = function(tmin, tmax) {
  if (length(tmin) != length(tmax)) {
    stop("`tmin` and `tmax` must have the same length.", call. = FALSE)
  }
  # Swapped columns are a common import mistake; fail loudly
  swapped = !is.na(tmin) & !is.na(tmax) & tmin > tmax
  if (any(swapped)) {
    stop(sprintf("%d day(s) have `tmin` above `tmax`.", sum(swapped)), call. = FALSE)
  }
  invisible(NULL)
}

#' Daily degree-days by the averaging method
#'
#' Subtracts the base temperature from the daily mean, after capping the
#' maximum at the upper threshold.
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @param base Lower developmental threshold in degrees Celsius.
#' @param upper Upper developmental threshold in degrees Celsius.
#' @return A numeric vector of daily degree-days.
#' @export
ddAverage = function(tmin, tmax, base = 10, upper = 30) {
  checkTemps(tmin, tmax)
  out = numeric(length(tmin))

  for (i in seq_along(tmin)) {
    if (is.na(tmin[i]) || is.na(tmax[i])) {
      out[i] = NA_real_
      next
    }
    # Horizontal cutoff: no extra development above the upper threshold
    hi = min(tmax[i], upper)
    lo = min(tmin[i], upper)
    out[i] = max(0, (hi + lo) / 2 - base)
  }
  out
}

#' Daily degree-days by the single-triangle method
#'
#' Approximates the daily temperature curve by a triangle and integrates the
#' area between the base and upper thresholds.
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @param base Lower developmental threshold in degrees Celsius.
#' @param upper Upper developmental threshold in degrees Celsius.
#' @return A numeric vector of daily degree-days.
#' @export
ddTriangle = function(tmin, tmax, base = 10, upper = 30) {
  checkTemps(tmin, tmax)
  out = numeric(length(tmin))

  for (i in seq_along(tmin)) {
    lo = tmin[i]
    hi = tmax[i]
    if (is.na(lo) || is.na(hi)) {
      out[i] = NA_real_
      next
    }
    if (lo >= upper) {
      out[i] = upper - base
    } else if (hi <= base) {
      out[i] = 0
    } else if (lo >= base) {
      out[i] = (lo + hi) / 2 - base
    } else {
      # Only the tip of the triangle above `base` contributes
      out[i] = (hi - base)^2 / (2 * (hi - lo))
    }

    # Remove the part of the triangle that pokes above `upper`
    if (hi > upper && lo < upper) {
      out[i] = out[i] - (hi - upper)^2 / (2 * (hi - lo))
    }
  }
  out
}

#' Daily degree-days by the single-sine method
#'
#' Approximates the daily temperature curve by a sine wave and integrates the
#' area between the base and upper thresholds.
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @param base Lower developmental threshold in degrees Celsius.
#' @param upper Upper developmental threshold in degrees Celsius.
#' @return A numeric vector of daily degree-days.
#' @export
ddSine = function(tmin, tmax, base = 10, upper = 30) {
  checkTemps(tmin, tmax)

  # Mean daily area of the sine curve above `threshold`
  areaAbove = function(mean, amp, threshold) {
    theta = asin((threshold - mean) / amp)
    ((mean - threshold) * (pi / 2 - theta) + amp * cos(theta)) / pi
  }

  out = numeric(length(tmin))
  for (i in seq_along(tmin)) {
    lo = tmin[i]
    hi = tmax[i]
    if (is.na(lo) || is.na(hi)) {
      out[i] = NA_real_
      next
    }

    mean = (hi + lo) / 2
    amp = (hi - lo) / 2
    if (lo >= upper) {
      out[i] = upper - base
    } else if (hi <= base) {
      out[i] = 0
    } else if (lo >= base) {
      out[i] = mean - base
    } else {
      out[i] = areaAbove(mean, amp, base)
    }
    if (hi > upper && lo < upper) {
      out[i] = out[i] - areaAbove(mean, amp, upper)
    }
  }
  out
}

#' Daily degree-days with a selectable method
#'
#' @param tmin,tmax Numeric vectors of daily minimum and maximum air
#'   temperature in degrees Celsius.
#' @param method One of `"average"`, `"triangle"` and `"sine"`.
#' @param ... Passed to the selected method, typically `base` and `upper`.
#' @return A numeric vector of daily degree-days.
#' @export
degreeDays = function(tmin, tmax, method = c("average", "triangle", "sine"), ...) {
  method = match.arg(method)
  switch(method,
    average = ddAverage(tmin, tmax, ...),
    triangle = ddTriangle(tmin, tmax, ...),
    sine = ddSine(tmin, tmax, ...)
  )
}

#' Accumulate degree-days from a biofix date
#'
#' Sums daily degree-days from a start date within each season, treating
#' missing days as zero but counting how many were missing.
#'
#' @param date A vector of class `Date`.
#' @param dd A numeric vector of daily degree-days matching `date`.
#' @param biofix Start of accumulation as a `"%m-%d"` string. Days before
#'   the biofix belong to the previous year's season.
#' @return A data frame with columns `date`, `season`, `dd`, `cumDd` and
#'   `nMissing`.
#' @export
accumulateDegreeDays = function(date, dd, biofix = "03-01") {
  if (length(date) != length(dd)) {
    stop("`date` and `dd` must have the same length.", call. = FALSE)
  }
  if (anyNA(date)) {
    stop("`date` must not contain missing values.", call. = FALSE)
  }

  yr = as.integer(format(date, "%Y"))
  start = as.Date(paste(yr, biofix, sep = "-"))
  season = ifelse(date >= start, yr, yr - 1L)

  # Walk in date order but write results back in the caller's row order
  ord = order(date)
  cumDd = numeric(length(dd))
  nMissing = integer(length(dd))
  total = 0
  missing = 0L
  for (k in seq_along(ord)) {
    i = ord[k]
    if (k > 1 && season[i] != season[ord[k - 1]]) {
      total = 0
      missing = 0L
    }
    if (is.na(dd[i])) {
      missing = missing + 1L
    } else {
      total = total + dd[i]
    }
    cumDd[i] = total
    nMissing[i] = missing
  }

  data.frame(date = date, season = season, dd = dd, cumDd = cumDd, nMissing = nMissing)
}

#' Heating and cooling degree-days
#'
#' Building-energy degree-days: the shortfall of the daily mean below the
#' balance point (heating) or its excess above it (cooling).
#'
#' @param tmean A numeric vector of daily mean air temperature in degrees
#'   Celsius.
#' @param base Balance-point temperature in degrees Celsius.
#' @return A numeric vector of daily degree-days.
#' @rdname buildingDegreeDays
#' @export
heatingDegreeDays = function(tmean, base = 15.5) {
  out = base - tmean
  out[!is.na(out) & out < 0] = 0
  out
}

#' @rdname buildingDegreeDays
#' @export
coolingDegreeDays = function(tmean, base = 22) {
  out = tmean - base
  out[!is.na(out) & out < 0] = 0
  out
}

#' Hourly chill units
#'
#' Converts hourly temperatures to chill units with a piecewise weighting in
#' which moderate cold counts fully and warm hours subtract chill.
#'
#' @param temp A numeric vector of hourly air temperature in degrees Celsius.
#' @param bands A data frame with columns `upper` and `weight`, sorted by
#'   `upper`; the last band applies to all warmer hours.
#' @return A numeric vector of hourly chill units.
#' @export
chillUnits = function(temp,
                      bands = data.frame(
                        upper = c(1.4, 2.4, 9.1, 12.4, 15.9, 18.0, Inf),
                        weight = c(0, 0.5, 1, 0.5, 0, -0.5, -1)
                      )) {
  out = numeric(length(temp))
  for (i in seq_along(temp)) {
    if (is.na(temp[i])) {
      out[i] = NA_real_
    } else {
      # First band whose upper edge is at or above the temperature
      out[i] = bands$weight[which(temp[i] <= bands$upper)[1]]
    }
  }
  out
}
