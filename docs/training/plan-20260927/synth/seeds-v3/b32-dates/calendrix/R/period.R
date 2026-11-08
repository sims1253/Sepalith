#' Number of days in the month of each date
#'
#' @param x A `Date` vector.
#' @return An integer vector of month lengths, one per element of `x`.
#' @keywords internal
days_in_month <- function(x) {
  lt <- as.POSIXlt(as.Date(x))
  year <- lt$year + 1900L
  mon <- lt$mon + 1L
  base <- c(31L, 28L, 31L, 30L, 31L, 30L, 31L, 31L, 30L, 31L, 30L, 31L)[mon]
  leap <- (year %% 4L == 0L & year %% 100L != 0L) | year %% 400L == 0L
  return(base + as.integer(mon == 2L & leap))
}

#' Shift dates by whole months
#'
#' Days that do not exist in the target month are clamped to its last day,
#' so 31 January plus one month is 28 or 29 February.
#'
#' @param x A `Date` vector.
#' @param months Integer number of months to shift by (recycled).
#' @return A `Date` vector the same length as `x`.
#' @keywords internal
shift_month <- function(x, months) {
  lt <- as.POSIXlt(as.Date(x))
  total <- (lt$year + 1900L) * 12L + lt$mon + as.integer(months)
  year <- total %/% 12L
  mon <- total %% 12L + 1L
  first <- as.Date(sprintf("%04d-%02d-01", year, mon))
  day <- pmin(lt$mday, days_in_month(first))
  return(first + (day - 1L))
}

#' Last day of the month of each date
#'
#' @param x A `Date` vector.
#' @return A `Date` vector the same length as `x`.
#' @keywords internal
month_end <- function(x) {
  x <- as.Date(x)
  return(x - as.POSIXlt(x)$mday + days_in_month(x))
}

#' Add a number of calendar periods to dates
#'
#' @param x A `Date` vector.
#' @param n Integer number of periods to add; negative values subtract.
#' @param unit One of `"day"`, `"week"`, `"fortnight"`, `"month"`,
#'   `"quarter"`, `"half"` or `"year"`.
#' @param keep_eom If `TRUE`, dates on the last day of a month stay on the
#'   last day of the target month for the month-based units.
#' @return A `Date` vector the same length as `x`.
#' @export
add_period <- function(x, n, unit = c("day", "week", "fortnight", "month", "quarter",
                                      "half", "year"),
                       keep_eom = FALSE) {
  unit <- match.arg(unit)
  x <- as.Date(x)
  if (!is.numeric(n) || anyNA(n)) {
    stop("`n` must be a numeric vector without missing values.", call. = FALSE)
  }
  n <- as.integer(n)
  eom <- keep_eom & !is.na(x) & x == month_end(x)

  out <- switch(unit,
    day = {
      x + n
    },
    week = {
      x + 7L * n
    },
    fortnight = {
      x + 14L * n
    },
    month = {
      out <- shift_month(x, n)
      out[eom] <- month_end(out[eom])
      out
    },
    quarter = {
      out <- shift_month(x, 3L * n)
      out[eom] <- month_end(out[eom])
      out
    },
    half = {
      out <- shift_month(x, 6L * n)
      out[eom] <- month_end(out[eom])
      out
    },
    year = {
      out <- shift_month(x, 12L * n)
      out[eom] <- month_end(out[eom])
      out
    }
  )
  return(out)
}

#' Count whole calendar periods between two dates
#'
#' A period only counts once it is complete: 31 January to 28 February
#' is one whole month (the end date is clamped), while 31 January to
#' 27 February is zero whole months but three whole weeks.
#'
#' @param from,to `Date` vectors of the same length.
#' @param unit One of `"day"`, `"week"`, `"fortnight"`, `"month"`,
#'   `"quarter"`, `"half"` or `"year"`.
#' @return An integer vector of signed period counts.
#' @export
periods_between <- function(from, to, unit = c("day", "week", "fortnight", "month",
                                               "quarter", "half", "year")) {
  unit <- match.arg(unit)
  from <- as.Date(from)
  to <- as.Date(to)
  days <- as.integer(to - from)
  sign <- ifelse(days < 0L, -1L, 1L)

  # Whole months from the earlier to the later date
  lo <- pmin(from, to)
  hi <- pmax(from, to)
  lo_lt <- as.POSIXlt(lo)
  hi_lt <- as.POSIXlt(hi)
  months <- (hi_lt$year - lo_lt$year) * 12L + (hi_lt$mon - lo_lt$mon)
  months <- months - as.integer(shift_month(lo, months) > hi)

  out <- switch(unit,
    day = {
      abs(days)
    },
    week = {
      abs(days) %/% 7L
    },
    fortnight = {
      abs(days) %/% 14L
    },
    month = {
      months
    },
    quarter = {
      months %/% 3L
    },
    half = {
      months %/% 6L
    },
    year = {
      months %/% 12L
    }
  )
  return(as.integer(sign * out))
}

#' Seconds per duration unit
#'
#' @format A named numeric vector.
#' @keywords internal
SECONDS_PER <- c(
  second = 1, minute = 60, hour = 3600, day = 86400, week = 604800
)

#' Convert a duration to a `difftime`
#'
#' @param n Numeric amount of time.
#' @param unit One of `"second"`, `"minute"`, `"hour"`, `"day"` or `"week"`.
#' @return A `difftime` object in the matching `difftime` units.
#' @export
as_period_difftime <- function(n, unit = c("second", "minute", "hour", "day", "week")) {
  unit <- match.arg(unit)
  secs <- n * SECONDS_PER[[unit]]
  units <- switch(unit,
    second = "secs",
    minute = "mins",
    hour = "hours",
    day = "days",
    week = "weeks"
  )
  out <- as.difftime(secs, units = "secs")
  units(out) <- units
  return(out)
}

#' Summarise the gap between two dates in several units
#'
#' @param from A single `Date`.
#' @param to A single `Date`.
#' @return A list with elements `from`, `to`, `direction` (`"forward"` or
#'   `"backward"`), `days`, `weeks`, `fortnights`, `months`, `quarters`,
#'   `halves` and `years`, each count computed by [periods_between()].
#' @export
period_summary <- function(from, to) {
  from <- as.Date(from)
  to <- as.Date(to)
  if (length(from) != 1L || is.na(from)) {
    stop("`from` must be a single non-missing date.", call. = FALSE)
  }
  if (length(to) != 1L || is.na(to)) {
    stop("`to` must be a single non-missing date.", call. = FALSE)
  }
  direction <- if (to >= from) "forward" else "backward"

  days <- periods_between(from, to, "day")
  weeks <- periods_between(from, to, "week")
  fortnights <- periods_between(from, to, "fortnight")
  months <- periods_between(from, to, "month")
  quarters <- periods_between(from, to, "quarter")
  halves <- periods_between(from, to, "half")
  years <- periods_between(from, to, "year")
  return(list(
    from = from,
    to = to,
    direction = direction,
    days = days,
    weeks = weeks,
    fortnights = fortnights,
    months = months,
    quarters = quarters,
    halves = halves,
    years = years
  ))
}

#' Validate arguments for period arithmetic
#'
#' @param x A `Date` vector.
#' @param n Integer number of periods.
#' @param unit Character scalar naming the unit.
#' @return `TRUE`, invisibly, or an error naming the first bad argument.
#' @noRd
check_period_args <- function(x, n, unit) {
  # Validate `x`
  if (!inherits(x, "Date")) {
    stop("`x` must be a `Date` vector.", call. = FALSE)
  }
  # Validate `n`
  if (!is.numeric(n) || any(n != round(n), na.rm = TRUE)) {
    stop("`n` must be a whole-number vector.", call. = FALSE)
  }
  # Validate `unit`
  if (!is.character(unit) || length(unit) != 1L) {
    stop("`unit` must be a single string.", call. = FALSE)
  }
  # Validate recycling of `x` and `n`
  if (length(n) != 1L && length(n) != length(x)) {
    stop("`n` must have length 1 or the length of `x`.", call. = FALSE)
  }
  return(invisible(TRUE))
}
