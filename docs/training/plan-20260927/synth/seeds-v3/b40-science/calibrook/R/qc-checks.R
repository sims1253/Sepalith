#' Quality-control limits for one measure
#'
#' @description
#' Plausible range, largest step between consecutive readings, and the
#' number of identical readings after which a series counts as stuck.
#'
#' @param measure One of `"temp"`, `"rh"`, `"pressure"`, `"wind_speed"`
#'   or `"precip"`.
#' @return A list with elements `lower`, `upper`, `maxStep` and `maxRun`.
#' @keywords internal
qcLimits = function(measure = c("temp", "rh", "pressure", "wind_speed", "precip")) {
  measure = match.arg(measure)
  limits = switch(measure,
    temp = list(lower = -50, upper = 50, maxStep = 5, maxRun = 6),
    rh = list(lower = 0, upper = 100, maxStep = 20, maxRun = 12),
    pressure = list(lower = 850, upper = 1080, maxStep = 3, maxRun = 24),
    wind_speed = list(lower = 0, upper = 60, maxStep = 15, maxRun = 6),
    precip = list(lower = 0, upper = 100, maxStep = 50, maxRun = Inf)
  )
  return(limits)
}

#' Range check
#'
#' @description
#' Flags readings outside `[lower, upper]`.
#'
#' @param x A numeric vector of readings in time order.
#' @param lower,upper The plausible range.
#' @return An integer vector of flags: `0` pass, `1` fail, `NA` where `x`
#'   is missing. Unless every reading is missing, the vector carries the
#'   attributes `check` (the check name) and `nFail`.
#' @family qc checks
#' @export
qcRange = function(x, lower, upper) {
  # Validate inputs
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }
  if (length(lower) != 1L || length(upper) != 1L || lower > upper) {
    stop("`lower` and `upper` must be single values with `lower <= upper`.", call. = FALSE)
  }
  if (all(is.na(x))) {
    return(rep(NA_integer_, length(x)))
  }

  # Flag failures
  flag = rep(0L, length(x))
  flag[x < lower | x > upper] = 1L

  # Mark missing readings and record the check
  flag[is.na(x)] = NA_integer_
  attr(flag, "check") = "range"
  attr(flag, "nFail") = sum(flag == 1L, na.rm = TRUE)
  return(flag)
}

#' Step check
#'
#' @description
#' Flags readings that differ from the previous non-missing reading by
#' more than `maxStep`.
#'
#' @param x A numeric vector of readings in time order.
#' @param maxStep The largest allowed change between consecutive readings.
#' @return An integer vector of flags: `0` pass, `1` fail, `NA` where `x`
#'   is missing. Unless every reading is missing, the vector carries the
#'   attributes `check` (the check name) and `nFail`.
#' @family qc checks
#' @export
qcStep = function(x, maxStep) {
  # Validate inputs
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }
  if (length(maxStep) != 1L || maxStep <= 0) {
    stop("`maxStep` must be a single positive value.", call. = FALSE)
  }
  if (all(is.na(x))) {
    return(rep(NA_integer_, length(x)))
  }

  # Flag failures
  flag = rep(0L, length(x))
  idx = which(!is.na(x))
  if (length(idx) > 1L) {
    jump = abs(diff(x[idx]))
    flag[idx[-1]][jump > maxStep] = 1L
  }

  # Mark missing readings and record the check
  flag[is.na(x)] = NA_integer_
  attr(flag, "check") = "step"
  attr(flag, "nFail") = sum(flag == 1L, na.rm = TRUE)
  return(flag)
}

#' Persistence check
#'
#' @description
#' Flags readings that belong to a run of more than `maxRun` identical
#' values, which usually means a stuck sensor.
#'
#' @param x A numeric vector of readings in time order.
#' @param maxRun The longest allowed run of identical readings.
#' @return An integer vector of flags: `0` pass, `1` fail, `NA` where `x`
#'   is missing. Unless every reading is missing, the vector carries the
#'   attributes `check` (the check name) and `nFail`.
#' @family qc checks
#' @export
qcPersistence = function(x, maxRun) {
  # Validate inputs
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }
  if (length(maxRun) != 1L || maxRun < 1) {
    stop("`maxRun` must be a single value of at least 1.", call. = FALSE)
  }
  if (all(is.na(x))) {
    return(rep(NA_integer_, length(x)))
  }

  # Flag failures
  flag = rep(0L, length(x))
  runs = rle(x)
  stuck = !is.na(runs$values) & runs$lengths > maxRun
  flag[inverse.rle(list(lengths = runs$lengths, values = stuck))] = 1L

  # Mark missing readings and record the check
  flag[is.na(x)] = NA_integer_
  attr(flag, "check") = "persistence"
  attr(flag, "nFail") = sum(flag == 1L, na.rm = TRUE)
  return(flag)
}

#' Spike check
#'
#' @description
#' Flags readings that differ from both neighbours by more than
#' `threshold` in the same direction.
#'
#' @param x A numeric vector of readings in time order.
#' @param threshold The smallest jump, up and back, that counts as a spike.
#' @return An integer vector of flags: `0` pass, `1` fail, `NA` where `x`
#'   is missing. Unless every reading is missing, the vector carries the
#'   attributes `check` (the check name) and `nFail`.
#' @family qc checks
#' @export
qcSpike = function(x, threshold) {
  # Validate inputs
  if (!is.numeric(x)) {
    stop("`x` must be a numeric vector.", call. = FALSE)
  }
  if (length(threshold) != 1L || threshold <= 0) {
    stop("`threshold` must be a single positive value.", call. = FALSE)
  }
  if (all(is.na(x))) {
    return(rep(NA_integer_, length(x)))
  }

  # Flag failures
  flag = rep(0L, length(x))
  n = length(x)
  if (n > 2L) {
    up = x[2:(n - 1)] - x[1:(n - 2)]
    down = x[2:(n - 1)] - x[3:n]
    spike = (up > threshold & down > threshold) | (up < -threshold & down < -threshold)
    flag[2:(n - 1)][!is.na(spike) & spike] = 1L
  }

  # Mark missing readings and record the check
  flag[is.na(x)] = NA_integer_
  attr(flag, "check") = "spike"
  attr(flag, "nFail") = sum(flag == 1L, na.rm = TRUE)
  return(flag)
}

#' Run range, step and persistence checks for one measure
#'
#' @description
#' Looks up the limits of `measure` with [qcLimits()] and runs
#' [qcRange()], [qcStep()] and [qcPersistence()] on `x`.
#'
#' @param x A numeric vector of readings in time order.
#' @param measure Passed to [qcLimits()].
#' @param limits Optional named list overriding some of the limits.
#' @return A data frame with columns `value`, `range`, `step`,
#'   `persistence` and `flag` (the largest of the three check flags).
#' @export
qcMeasure = function(x, measure, limits = list()) {
  lim = utils::modifyList(qcLimits(measure), limits)

  # Range
  range = qcRange(x, lim$lower, lim$upper)

  # Step
  step = qcStep(x, lim$maxStep)

  # Persistence
  persistence = if (is.finite(lim$maxRun)) {
    qcPersistence(x, lim$maxRun)
  } else {
    ifelse(is.na(x), NA_integer_, 0L)
  }

  flag = pmax(range, step, persistence)
  return(data.frame(
    value = x,
    range = range,
    step = step,
    persistence = persistence,
    flag = flag
  ))
}
