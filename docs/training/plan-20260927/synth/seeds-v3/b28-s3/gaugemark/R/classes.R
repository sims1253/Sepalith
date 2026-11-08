#' @title Conversion factor to metres
#' @param unit One of `"mm"`, `"cm"`, `"m"`, `"km"` or `"in"`.
#' @return The number of metres in one `unit`.
#' @keywords internal
unitFactor = function(unit = c("mm", "cm", "m", "km", "in")) {
  unit = match.arg(unit)
  switch(unit,
    mm = 1e-3,
    cm = 1e-2,
    m = 1,
    km = 1e3,
    `in` = 0.0254
  )
}

#' @title Low-level reading constructor
#' @description Wraps a double vector and a `unit` attribute without checks.
#' @param value A double vector of measured lengths.
#' @param unit A single unit string.
#' @return An object of class `gm_reading`.
#' @keywords internal
newReading = function(value = double(), unit = "m") {
  stopifnot(is.double(value), is.character(unit))
  structure(value, unit = unit, class = "gm_reading")
}

#' @title Validate a reading
#' @description Checks the `unit` attribute, then the values: they must be
#'   doubles, finite where present, and not negative.
#' @param x An object of class `gm_reading`.
#' @return `x`, unchanged, if it is valid.
#' @keywords internal
validateReading = function(x) {
  value = unclass(x)
  unit = attr(x, "unit")

  # Check the unit first: every later step depends on it
  if (!is.character(unit) || length(unit) != 1L) {
    stop("`unit` must be a single string.", call. = FALSE)
  }
  if (!unit %in% c("mm", "cm", "m", "km", "in")) {
    stop(sprintf("Unknown unit '%s'.", unit), call. = FALSE)
  }

  # Check the values
  if (!is.double(value)) {
    stop("`value` must be a double vector.", call. = FALSE)
  }
  if (any(is.infinite(value))) {
    stop("`value` must not contain infinite values.", call. = FALSE)
  }
  if (any(value < 0, na.rm = TRUE)) {
    stop("`value` must not be negative.", call. = FALSE)
  }
  x
}

#' @title Create length readings
#' @param value A numeric vector of measured lengths, or a `gm_reading`
#'   which is converted to `unit`.
#' @param unit The unit of the result.
#' @return A validated object of class `gm_reading`.
#' @export
reading = function(value, unit = c("m", "mm", "cm", "km", "in")) {
  unit = match.arg(unit)
  # An existing reading is converted rather than re-wrapped
  if (inherits(value, "gm_reading")) {
    ratio = unitFactor(attr(value, "unit")) / unitFactor(unit)
    value = as.double(unclass(value)) * ratio
  }
  if (!is.numeric(value)) {
    stop(sprintf("`value` must be numeric, not <%s>.", class(value)[1L]), call. = FALSE)
  }
  value = as.double(value)
  x = newReading(value, unit)
  validateReading(x)
}

#' @title Low-level range constructor
#' @description Wraps `lower` and `upper` bounds and a `unit` attribute
#'   without checks.
#' @param lower A double vector of lower bounds.
#' @param upper A double vector of upper bounds.
#' @param unit A single unit string.
#' @return An object of class `gm_range`.
#' @keywords internal
newRange = function(lower = double(), upper = double(), unit = "m") {
  stopifnot(is.double(lower), is.double(upper), is.character(unit))
  structure(list(lower = lower, upper = upper), unit = unit, class = "gm_range")
}

#' @title Validate a range
#' @description Checks the `unit` attribute, then `lower` and `upper` in
#'   turn with the same rules as [validateReading()], then that no range has
#'   `upper` below `lower`.
#' @param x An object of class `gm_range`.
#' @return `x`, unchanged, if it is valid.
#' @keywords internal
validateRange = function(x) {
  fields = unclass(x)
  unit = attr(x, "unit")

  # Check the unit first: every later step depends on it
  if (!is.character(unit) || length(unit) != 1L) {
    stop("`unit` must be a single string.", call. = FALSE)
  }
  if (!unit %in% c("mm", "cm", "m", "km", "in")) {
    stop(sprintf("Unknown unit '%s'.", unit), call. = FALSE)
  }

  # Check `lower`
  if (!is.double(fields$lower)) {
    stop("`lower` must be a double vector.", call. = FALSE)
  }
  if (any(is.infinite(fields$lower))) {
    stop("`lower` must not contain infinite values.", call. = FALSE)
  }
  if (any(fields$lower < 0, na.rm = TRUE)) {
    stop("`lower` must not be negative.", call. = FALSE)
  }

  # Check `upper`
  if (!is.double(fields$upper)) {
    stop("`upper` must be a double vector.", call. = FALSE)
  }
  if (any(is.infinite(fields$upper))) {
    stop("`upper` must not contain infinite values.", call. = FALSE)
  }
  if (any(fields$upper < 0, na.rm = TRUE)) {
    stop("`upper` must not be negative.", call. = FALSE)
  }
  if (length(fields$upper) != length(fields$lower)) {
    stop("`upper` must have the same length as `lower`.", call. = FALSE)
  }

  # Check that every range is ordered
  bad = which(fields$upper < fields$lower)
  if (length(bad) > 0L) {
    stop(sprintf("Range %d has `upper` below `lower`.", bad[1L]), call. = FALSE)
  }
  x
}

#' @title Create length ranges
#' @description Length-one `lower` or `upper` are recycled to the common
#'   length.
#' @param lower A numeric vector of lower bounds, or a `gm_reading`.
#' @param upper A numeric vector of upper bounds, or a `gm_reading`.
#' @param unit The unit of the result; readings are converted to it.
#' @return A validated object of class `gm_range`.
#' @export
gaugeRange = function(lower, upper, unit = c("m", "mm", "cm", "km", "in")) {
  unit = match.arg(unit)

  # Check `lower`
  if (inherits(lower, "gm_reading")) {
    ratio = unitFactor(attr(lower, "unit")) / unitFactor(unit)
    lower = as.double(unclass(lower)) * ratio
  }
  if (!is.numeric(lower) || length(lower) == 0L) {
    stop("`lower` must be a non-empty numeric vector.", call. = FALSE)
  }

  # Check `upper`
  if (inherits(upper, "gm_reading")) {
    ratio = unitFactor(attr(upper, "unit")) / unitFactor(unit)
    upper = as.double(unclass(upper)) * ratio
  }
  if (!is.numeric(upper) || length(upper) == 0L) {
    stop("`upper` must be a non-empty numeric vector.", call. = FALSE)
  }

  lower = as.double(lower)
  upper = as.double(upper)
  n = max(length(lower), length(upper))

  # Recycle length-one bounds
  if (length(lower) == 1L) {
    lower = rep(lower, n)
  }
  if (length(upper) == 1L) {
    upper = rep(upper, n)
  }
  x = newRange(lower, upper, unit)
  validateRange(x)
}
