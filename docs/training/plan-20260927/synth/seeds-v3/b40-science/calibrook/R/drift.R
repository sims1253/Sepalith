#' Validate inputs to the drift functions
#'
#' @param x A numeric vector of readings in time order.
#' @param time A numeric or `POSIXct` vector of reading times.
#' @return `NULL`, invisibly; called for its errors.
#' @noRd
checkDriftInputs = function(x, time) {
  # Check `x`
  if (!is.numeric(x)) {
    stop("`x` must be numeric.", call. = FALSE)
  }
  if (length(x) < 2L) {
    stop("`x` must have at least 2 values.", call. = FALSE)
  }

  # Check `time`
  if (!is.numeric(time) && !inherits(time, "POSIXct")) {
    stop("`time` must be numeric or POSIXct.", call. = FALSE)
  }
  if (length(time) != length(x)) {
    stop("`time` must have the same length as `x`.", call. = FALSE)
  }
  if (anyNA(time)) {
    stop("`time` must not contain missing values.", call. = FALSE)
  }
  if (is.unsorted(time)) {
    stop("`time` must be sorted.", call. = FALSE)
  }
  return(invisible(NULL))
}

#' Remove instrument drift between two check visits
#'
#' @description
#' At the start and the end of a deployment the sensor was compared with
#' a standard; `startOffset` and `endOffset` are the sensor errors found
#' (sensor minus standard). The error in between is modelled with
#' `method` and subtracted.
#'
#' @param x A numeric vector of readings in time order.
#' @param time A numeric or `POSIXct` vector of reading times.
#' @param startOffset,endOffset Sensor error at the first and last reading.
#' @param method `"linear"` interpolates the error, `"step"` keeps the
#'   start error until the midpoint and the end error after it,
#'   `"exponential"` lets the error approach `endOffset` with a time
#'   constant of a third of the deployment, and `"end"` applies
#'   `endOffset` throughout.
#' @return The corrected readings.
#' @export
driftCorrect = function(x, time, startOffset, endOffset,
                        method = c("linear", "step", "exponential", "end")) {
  method = match.arg(method)
  checkDriftInputs(x, time)
  t = as.numeric(time)
  frac = (t - t[1]) / (t[length(t)] - t[1])
  delta = endOffset - startOffset

  error = switch(method,
    linear = {
      startOffset + delta * frac
    },
    step = {
      ifelse(frac < 0.5, startOffset, endOffset)
    },
    exponential = {
      shape = (1 - exp(-3 * frac)) / (1 - exp(-3))
      startOffset + delta * shape
    },
    end = {
      rep(endOffset, length(x))
    }
  )
  return(x - error)
}

#' Rate of drift in readings per unit of time
#'
#' @param x A numeric vector of readings in time order.
#' @param time A `POSIXct` vector of reading times.
#' @param unit The time unit of the returned rate: `"hour"`, `"day"`,
#'   `"week"`, `"month"` (30 days) or `"year"` (365 days).
#' @return The least-squares slope of `x` on `time`, per `unit`.
#' @export
driftRate = function(x, time, unit = c("day", "hour", "week", "month", "year")) {
  unit = match.arg(unit)
  checkDriftInputs(x, time)
  seconds = c(hour = 3600, day = 24 * 3600, week = 7 * 24 * 3600,
              month = 30 * 24 * 3600, year = 365 * 24 * 3600)
  t = as.numeric(time) / seconds[[unit]]
  ok = is.finite(x) & is.finite(t)
  slope = stats::cov(t[ok], x[ok]) / stats::var(t[ok])
  return(slope)
}

#' Evaluate a zero and span check
#'
#' @description
#' The analyser is given a zero gas and a span gas of known value. Each
#' reading is compared with its target and tolerance, and the two readings
#' together give a corrected gain and offset.
#'
#' @param zero Reading on the zero gas.
#' @param span Reading on the span gas.
#' @param spanTarget Known value of the span gas.
#' @param zeroTol Allowed absolute error on the zero gas.
#' @param spanTol Allowed relative error on the span gas, as a fraction.
#' @return A list with elements `zeroError`, `zeroPass`, `spanError`,
#'   `spanPass`, `gain` and `offset`.
#' @export
zeroSpanCheck = function(zero, span, spanTarget, zeroTol = 1, spanTol = 0.05) {
  if (span <= zero) {
    stop("`span` must be greater than `zero`.", call. = FALSE)
  }

  # Zero
  zeroError = zero - 0
  zeroPass = abs(zeroError) <= zeroTol

  # Span
  spanError = (span - spanTarget) / spanTarget
  spanPass = abs(spanError) <= spanTol

  gain = spanTarget / (span - zero)
  offset = -gain * zero
  return(list(
    zeroError = zeroError,
    zeroPass = zeroPass,
    spanError = spanError,
    spanPass = spanPass,
    gain = gain,
    offset = offset
  ))
}

#' Summarise drift for several sensors
#'
#' @param data A data frame with a `time` column and one column per sensor.
#' @param sensors Sensor columns to summarise.
#' @param unit Passed to [driftRate()].
#' @return A data frame with columns `sensor`, `first`, `last`, `change`,
#'   `span` (days between the first and last valid reading) and `rate`,
#'   one row per sensor.
#' @export
driftSummary = function(data, sensors, unit = "day") {
  missingCols = setdiff(c("time", sensors), names(data))
  if (length(missingCols) > 0L) {
    stop(sprintf("`data` lacks columns: %s.", paste(missingCols, collapse = ", ")),
         call. = FALSE)
  }
  out = data.frame(sensor = sensors, first = NA_real_, last = NA_real_,
                   change = NA_real_, span = NA_real_, rate = NA_real_)
  for (i in seq_along(sensors)) {
    x = data[[sensors[i]]]
    ok = which(is.finite(x))
    if (length(ok) < 2L) {
      next
    }
    out$first[i] = x[ok[1]]
    out$last[i] = x[ok[length(ok)]]
    out$change[i] = out$last[i] - out$first[i]
    out$span[i] = as.numeric(difftime(data$time[ok[length(ok)]], data$time[ok[1]],
                                      units = "days"))
    out$rate[i] = driftRate(x, data$time, unit)
  }
  return(out)
}
