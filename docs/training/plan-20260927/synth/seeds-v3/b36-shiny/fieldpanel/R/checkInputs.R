#' Parse a count typed into a text input
#'
#' @description
#' Trims the raw text, converts it to an integer and checks its range.
#'
#' @param x Raw text from the input.
#' @param label Label of the input, used in messages.
#' @param min,max Smallest and largest accepted value.
#' @return A list with elements `value` (`NULL` on failure) and `error`
#'   (`NULL` on success).
#' @family input parsers
#' @export
parseCountInput = function(x, label, min = 0, max = Inf) {
  x = trimws(as.character(x))
  if (length(x) != 1L || is.na(x) || !nzchar(x)) {
    return(list(value = NULL, error = sprintf("%s is required.", label)))
  }
  value = suppressWarnings(as.numeric(x))
  if (is.na(value) || value != round(value)) {
    return(list(value = NULL, error = sprintf("%s must be a whole number.", label)))
  }
  if (value < min || value > max) {
    return(list(value = NULL, error = sprintf("%s must be between %s and %s.",
                                              label, min, max)))
  }
  return(list(value = as.integer(value), error = NULL))
}

#' Parse a rate typed into a text input
#'
#' @description
#' Trims the raw text, converts it to a proportion and checks its range.
#' A trailing `%` divides the number by 100.
#'
#' @param x Raw text from the input.
#' @param label Label of the input, used in messages.
#' @param min,max Smallest and largest accepted value.
#' @return A list with elements `value` (`NULL` on failure) and `error`
#'   (`NULL` on success).
#' @family input parsers
#' @export
parseRateInput = function(x, label, min = 0, max = 1) {
  x = trimws(as.character(x))
  if (length(x) != 1L || is.na(x) || !nzchar(x)) {
    return(list(value = NULL, error = sprintf("%s is required.", label)))
  }
  isPercent = endsWith(x, "%")
  value = suppressWarnings(as.numeric(sub("%$", "", x)))
  if (is.na(value)) {
    return(list(value = NULL, error = sprintf("%s must be a number.", label)))
  }
  if (isPercent) {
    value = value / 100
  }
  if (value < min || value > max) {
    return(list(value = NULL, error = sprintf("%s must be between %s and %s.",
                                              label, min, max)))
  }
  return(list(value = value, error = NULL))
}

#' Parse a date typed into a text input
#'
#' @description
#' Trims the raw text, converts it to a `Date` and checks its range.
#'
#' @param x Raw text from the input, in `YYYY-MM-DD` form.
#' @param label Label of the input, used in messages.
#' @param min,max Earliest and latest accepted date.
#' @return A list with elements `value` (`NULL` on failure) and `error`
#'   (`NULL` on success).
#' @family input parsers
#' @export
parseDateInput = function(x, label, min = as.Date("2000-01-01"),
                          max = as.Date("2099-12-31")) {
  x = trimws(as.character(x))
  if (length(x) != 1L || is.na(x) || !nzchar(x)) {
    return(list(value = NULL, error = sprintf("%s is required.", label)))
  }
  value = as.Date(x, format = "%Y-%m-%d", optional = TRUE)
  if (is.na(value)) {
    return(list(value = NULL, error = sprintf("%s must be a date (YYYY-MM-DD).", label)))
  }
  if (value < min || value > max) {
    return(list(value = NULL, error = sprintf("%s must be between %s and %s.",
                                              label, min, max)))
  }
  return(list(value = value, error = NULL))
}

#' Parse an identifier typed into a text input
#'
#' @description
#' Trims the raw text, converts it to upper case and checks it against a
#' pattern.
#'
#' @param x Raw text from the input.
#' @param label Label of the input, used in messages.
#' @param pattern Regular expression the identifier must match.
#' @return A list with elements `value` (`NULL` on failure) and `error`
#'   (`NULL` on success).
#' @family input parsers
#' @export
parseIdInput = function(x, label, pattern = "^[A-Z]{2}[0-9]{3}$") {
  x = trimws(as.character(x))
  if (length(x) != 1L || is.na(x) || !nzchar(x)) {
    return(list(value = NULL, error = sprintf("%s is required.", label)))
  }
  value = toupper(x)
  if (!grepl(pattern, value)) {
    return(list(value = NULL, error = sprintf("%s must match %s.", label, pattern)))
  }
  return(list(value = value, error = NULL))
}

#' Check the settings of a fieldwork wave
#'
#' @description
#' Parses every field of the wave form and stops at nothing: all errors are
#' collected so the form can show them together.
#'
#' @param waveId Raw text of the wave identifier.
#' @param startDate Raw text of the first fieldwork day.
#' @param endDate Raw text of the last fieldwork day.
#' @param targetN Raw text of the target number of interviews.
#' @param responseRate Raw text of the expected response rate.
#' @return A list with elements `values` (a named list of parsed values) and
#'   `errors` (a named character vector, empty when the form is valid).
#' @export
checkWaveSettings = function(waveId, startDate, endDate, targetN, responseRate) {
  values = list()
  errors = character(0)

  # Parse `waveId`
  parsed = parseIdInput(waveId, "Wave ID")
  if (is.null(parsed$error)) {
    values$waveId = parsed$value
  } else {
    errors[["waveId"]] = parsed$error
  }

  # Parse `startDate`
  parsed = parseDateInput(startDate, "Start date")
  if (is.null(parsed$error)) {
    values$startDate = parsed$value
  } else {
    errors[["startDate"]] = parsed$error
  }

  # Parse `endDate`
  parsed = parseDateInput(endDate, "End date")
  if (is.null(parsed$error)) {
    values$endDate = parsed$value
  } else {
    errors[["endDate"]] = parsed$error
  }

  # Parse `targetN`
  parsed = parseCountInput(targetN, "Target interviews", min = 1)
  if (is.null(parsed$error)) {
    values$targetN = parsed$value
  } else {
    errors[["targetN"]] = parsed$error
  }

  # Parse `responseRate`
  parsed = parseRateInput(responseRate, "Response rate")
  if (is.null(parsed$error)) {
    values$responseRate = parsed$value
  } else {
    errors[["responseRate"]] = parsed$error
  }

  if (!is.null(values$startDate) && !is.null(values$endDate) &&
      values$endDate < values$startDate) {
    errors[["endDate"]] = "End date must not be before Start date."
  }
  return(list(values = values, errors = errors))
}

#' Check the contact settings of a fieldwork wave
#'
#' @description
#' Parses every field of the contact form, using the limits in `limits`,
#' and collects all errors like [checkWaveSettings()].
#'
#' @param maxAttempts Raw text of the largest number of contact attempts.
#' @param minGapHours Raw text of the minimum hours between attempts.
#' @param windowStart Raw text of the first hour of the call window.
#' @param windowEnd Raw text of the last hour of the call window.
#' @return A list with elements `values` (a named list of parsed values) and
#'   `errors` (a named character vector, empty when the form is valid).
#' @export
checkContactSettings = function(maxAttempts, minGapHours, windowStart, windowEnd) {
  labels = c(maxAttempts = "Max attempts", minGapHours = "Minimum gap",
             windowStart = "Call window start", windowEnd = "Call window end")
  limits = list(maxAttempts = c(1, 20), minGapHours = c(0, 168),
                windowStart = c(0, 23), windowEnd = c(1, 24))
  values = list()
  errors = character(0)

  # Parse `maxAttempts`
  parsed = parseCountInput(maxAttempts, labels[["maxAttempts"]],
                           min = limits$maxAttempts[1], max = limits$maxAttempts[2])
  if (is.null(parsed$error)) {
    values$maxAttempts = parsed$value
  } else {
    errors[["maxAttempts"]] = parsed$error
  }

  # Parse `minGapHours`
  parsed = parseCountInput(minGapHours, labels[["minGapHours"]],
                           min = limits$minGapHours[1], max = limits$minGapHours[2])
  if (is.null(parsed$error)) {
    values$minGapHours = parsed$value
  } else {
    errors[["minGapHours"]] = parsed$error
  }

  # Parse `windowStart`
  parsed = parseCountInput(windowStart, labels[["windowStart"]],
                           min = limits$windowStart[1], max = limits$windowStart[2])
  if (is.null(parsed$error)) {
    values$windowStart = parsed$value
  } else {
    errors[["windowStart"]] = parsed$error
  }

  # Parse `windowEnd`
  parsed = parseCountInput(windowEnd, labels[["windowEnd"]],
                           min = limits$windowEnd[1], max = limits$windowEnd[2])
  if (is.null(parsed$error)) {
    values$windowEnd = parsed$value
  } else {
    errors[["windowEnd"]] = parsed$error
  }

  if (!is.null(values$windowStart) && !is.null(values$windowEnd) &&
      values$windowEnd <= values$windowStart) {
    errors[["windowEnd"]] = "Call window end must be after Call window start."
  }
  return(list(values = values, errors = errors))
}

#' Check an uploaded quota table
#'
#' @description
#' A quota table has one row per cell with columns `cell`, `target`,
#' `achieved` and `weight`.
#'
#' @param quota A data frame read from the uploaded file.
#' @return A character vector of problems, empty when the table is valid.
#' @export
checkQuotaTable = function(quota) {
  columns = c("cell", "target", "achieved", "weight")
  missing = setdiff(columns, names(quota))
  if (length(missing) > 0L) {
    return(sprintf("Missing column: %s.", missing))
  }
  problems = character(0)

  # Check `cell`
  if (anyNA(quota$cell) || anyDuplicated(quota$cell) > 0L) {
    problems = c(problems, "Column cell must be unique and complete.")
  }

  # Check `target`
  if (!is.numeric(quota$target) || any(quota$target < 0, na.rm = TRUE)) {
    problems = c(problems, "Column target must be non-negative numbers.")
  }

  # Check `achieved`
  if (!is.numeric(quota$achieved) || any(quota$achieved < 0, na.rm = TRUE)) {
    problems = c(problems, "Column achieved must be non-negative numbers.")
  }

  # Check `weight`
  if (!is.numeric(quota$weight) || any(quota$weight < 0, na.rm = TRUE)) {
    problems = c(problems, "Column weight must be non-negative numbers.")
  }
  return(problems)
}
