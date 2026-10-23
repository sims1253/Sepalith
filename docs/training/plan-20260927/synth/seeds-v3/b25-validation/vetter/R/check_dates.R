#' Parse dates without silent failures
#'
#' @description
#' Converts `x` to `Date`, trying each of `formats` in turn for character
#' input, and fails if any non-missing element could not be parsed.
#'
#' @param x A `Date`, `POSIXct` or character vector.
#' @param name Argument name used in error messages.
#' @param formats Candidate formats for character input.
#' @return A `Date` vector the same length as `x`.
#' @keywords internal
as_date_strict <- function(x, name = "x", formats = c("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y")) {
  if (inherits(x, "Date")) {
    return(x)
  }
  if (inherits(x, "POSIXct")) {
    return(as.Date(x))
  }
  if (!is.character(x)) {
    stop(sprintf("`%s` must be a date or a string, not %s.", name, class(x)[1]),
         call. = FALSE)
  }
  out <- rep(as.Date(NA), length(x))
  for (fmt in formats) {
    todo <- is.na(out) & !is.na(x)
    out[todo] <- as.Date(x[todo], format = fmt)
  }
  bad <- which(is.na(out) & !is.na(x))
  if (length(bad) > 0L) {
    stop(sprintf("`%s` has unparseable date(s): %s.", name, paste(x[bad], collapse = ", ")),
         call. = FALSE)
  }
  out
}

#' Check a single date
#'
#' @param x Value to check, coercible with [as_date_strict()].
#' @param name Argument name used in error messages.
#' @param min,max Inclusive bounds, `Date` or `NULL` for unbounded.
#' @param allow_na Whether a missing date is accepted.
#' @return `x` as a `Date`.
#' @export
check_date <- function(x, name = "x", min = NULL, max = NULL, allow_na = FALSE) {
  if (length(x) != 1L) {
    stop(sprintf("`%s` must be a single date.", name), call. = FALSE)
  }
  x <- as_date_strict(x, name = name)
  if (allow_na && is.na(x)) {
    return(x)
  }
  if (is.na(x)) {
    stop(sprintf("`%s` must not be missing.", name), call. = FALSE)
  }
  if (!is.null(min) && x < min) {
    stop(sprintf("`%s` must be on or after %s.", name, format(min)), call. = FALSE)
  }
  if (!is.null(max) && x > max) {
    stop(sprintf("`%s` must be on or before %s.", name, format(max)), call. = FALSE)
  }
  x
}

#' Check a date range
#'
#' @description
#' Validates `from` and `to` as single dates, fills an open end with
#' `default_from` or `default_to`, and checks their order.
#'
#' @param from,to Range limits, coercible with [as_date_strict()], or
#'   `NULL` for an open end.
#' @param default_from,default_to Dates used for open ends.
#' @param max_days Longest allowed range in days.
#' @return A list with elements `from`, `to` and `days`.
#' @export
check_date_range <- function(from, to, default_from = as.Date("1900-01-01"),
                             default_to = Sys.Date(), max_days = Inf) {
  if (is.null(from)) {
    from <- default_from
  } else {
    from <- check_date(from, name = "from")
  }
  if (is.null(to)) {
    to <- default_to
  } else {
    to <- check_date(to, name = "to")
  }
  if (to < from) {
    stop("`to` must not be before `from`.", call. = FALSE)
  }
  days <- as.integer(to - from)
  if (days > max_days) {
    stop(sprintf("The range spans %d days, more than `max_days` (%s).", days,
                 format(max_days)), call. = FALSE)
  }
  list(from = from, to = to, days = days)
}

#' Check a reporting period
#'
#' @param period One of `"day"`, `"week"`, `"fortnight"`, `"month"`,
#'   `"quarter"` or `"year"`.
#' @param n Number of periods, a positive whole number.
#' @return A list with elements `period`, `n`, `by` (a `seq()` step) and
#'   `approx_days`.
#' @export
check_period <- function(period = c("day", "week", "fortnight", "month", "quarter", "year"),
                         n = 1L) {
  period <- match.arg(period)
  if (!is.numeric(n) || length(n) != 1L || is.na(n) || n < 1 || n != round(n)) {
    stop("`n` must be a positive whole number.", call. = FALSE)
  }
  n <- as.integer(n)
  switch(period,
    day = {
      by <- paste(n, "days")
      approx_days <- n
    },
    week = {
      by <- paste(n, "weeks")
      approx_days <- 7L * n
    },
    fortnight = {
      by <- paste(2L * n, "weeks")
      approx_days <- 14L * n
    },
    month = {
      by <- paste(n, "months")
      approx_days <- 30L * n
    },
    quarter = {
      by <- paste(3L * n, "months")
      approx_days <- 91L * n
    },
    year = {
      by <- paste(n, "years")
      approx_days <- 365L * n
    }
  )
  list(period = period, n = n, by = by, approx_days = approx_days)
}

#' Check a timestamp vector
#'
#' @param x A `POSIXct` vector or character in ISO 8601 format.
#' @param name Argument name used in error messages.
#' @param tz Time zone used to parse character input.
#' @param allow_na Whether missing values are accepted.
#' @return `x` as `POSIXct`.
#' @export
check_timestamp <- function(x, name = "x", tz = "UTC", allow_na = FALSE) {
  if (is.character(x)) {
    parsed <- as.POSIXct(x, tz = tz, tryFormats = c("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"),
                         optional = TRUE)
    bad <- which(is.na(parsed) & !is.na(x))
    if (length(bad) > 0L) {
      stop(sprintf("`%s` has unparseable timestamp(s): %s.", name,
                   paste(x[bad], collapse = ", ")), call. = FALSE)
    }
    x <- parsed
  }
  if (!inherits(x, "POSIXct")) {
    stop(sprintf("`%s` must be a timestamp or a string, not %s.", name, class(x)[1]),
         call. = FALSE)
  }
  if (!allow_na && anyNA(x)) {
    stop(sprintf("`%s` must not be missing.", name), call. = FALSE)
  }
  x
}

#' Check the arguments of a schedule builder
#'
#' @description
#' Runs the date, period and time zone checks in the order the schedule
#' builder needs them.
#'
#' @param start_date,end_date Schedule limits; `end_date` may be `NULL`.
#' @param period Period name passed to [check_period()].
#' @param every Number of periods between events.
#' @param tz Time zone name.
#' @return A list with elements `from`, `to`, `period`, `by` and `tz`.
#' @keywords internal
check_schedule_args <- function(start_date, end_date, period, every, tz) {
  # 1. Validate `start_date` and `end_date` together
  range <- check_date_range(start_date, end_date, default_to = Sys.Date())

  # 2. Validate `period` and `every`
  spec <- check_period(period, n = every)
  if (spec$approx_days > range$days) {
    stop(sprintf("`every` %s spans more than the date range (%d days).", spec$period,
                 range$days), call. = FALSE)
  }

  # 3. Validate `tz`
  if (!is.character(tz) || length(tz) != 1L || !tz %in% OlsonNames()) {
    stop("`tz` must be a single known time zone name.", call. = FALSE)
  }

  # 4. Assemble the result
  list(from = range$from, to = range$to, period = spec$period, by = spec$by, tz = tz)
}

#' Check the dates of an event record
#'
#' @description
#' Validates the four dates of an event record and their order: an event
#' must be created before it starts, start before it ends, and be updated
#' no earlier than it was created.
#'
#' @param start,end Event start and end, coercible with [as_date_strict()].
#' @param created,updated Record creation and update dates; `updated` may
#'   be `NULL`, in which case `created` is used.
#' @return A list with elements `start`, `end`, `created`, `updated` and
#'   `duration` (in days).
#' @export
check_event_dates <- function(start, end, created, updated = NULL) {
  # Validate each date
  start <- check_date(start, name = "start")
  end <- check_date(end, name = "end")
  created <- check_date(created, name = "created")
  if (is.null(updated)) {
    updated <- created
  } else {
    updated <- check_date(updated, name = "updated")
  }

  # Validate the order of `start` and `end`
  if (end < start) {
    stop("`end` must not be before `start`.", call. = FALSE)
  }

  # Validate the order of `created` and `start`
  if (start < created) {
    stop("`start` must not be before `created`.", call. = FALSE)
  }

  # Validate the order of `created` and `updated`
  if (updated < created) {
    stop("`updated` must not be before `created`.", call. = FALSE)
  }

  list(
    start = start,
    end = end,
    created = created,
    updated = updated,
    duration = as.integer(end - start)
  )
}
