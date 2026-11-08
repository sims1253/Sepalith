#' Day count between two dates
#'
#' @details
#' * `"actual"`: calendar days;
#' * `"thirty360"`: US bond basis, a start day of 31 becomes 30, and an end
#'   day of 31 becomes 30 when the start day is 30 or 31;
#' * `"thirtyE360"`: Eurobond basis, both days of 31 become 30;
#' * `"thirtyEPlus360"`: an end day of 31 rolls to the 1st of the next
#'   month instead.
#'
#' @param start,end `Date` vectors of equal length.
#' @param convention One of `"actual"`, `"thirty360"`, `"thirtyE360"` or
#'   `"thirtyEPlus360"`.
#' @return A numeric vector of day counts.
#' @keywords internal
dayCount = function(start, end,
                    convention = c("actual", "thirty360", "thirtyE360", "thirtyEPlus360")) {
  convention = match.arg(convention)
  a = as.POSIXlt(as.Date(start))
  b = as.POSIXlt(as.Date(end))
  y1 = a$year + 1900L
  y2 = b$year + 1900L
  m1 = a$mon + 1L
  m2 = b$mon + 1L

  switch(convention,
    actual = {
      as.numeric(as.Date(end) - as.Date(start))
    },
    thirty360 = {
      d1 = pmin(a$mday, 30L)
      d2 = ifelse(d1 == 30L, pmin(b$mday, 30L), b$mday)
      360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    },
    thirtyE360 = {
      d1 = pmin(a$mday, 30L)
      d2 = pmin(b$mday, 30L)
      360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    },
    thirtyEPlus360 = {
      d1 = pmin(a$mday, 30L)
      roll = b$mday == 31L
      d2 = ifelse(roll, 1L, b$mday)
      m2 = m2 + roll
      360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    }
  )
}

#' Actual/actual year fraction
#'
#' Days falling in a leap year are divided by 366, all other days by 365.
#'
#' @param start,end `Date` vectors of equal length.
#' @return A numeric vector of year fractions.
#' @noRd
actActFraction = function(start, end) {
  start = as.Date(start)
  end = as.Date(end)
  vapply(seq_along(start), function(i) {
    years = seq(as.POSIXlt(start[i])$year, as.POSIXlt(end[i])$year) + 1900L
    total = 0
    for (y in years) {
      from = max(start[i], as.Date(sprintf("%d-01-01", y)))
      to = min(end[i], as.Date(sprintf("%d-01-01", y + 1L)))
      leap = (y %% 4L == 0L & y %% 100L != 0L) | y %% 400L == 0L
      total = total + as.numeric(to - from) / ifelse(leap, 366, 365)
    }
    total
  }, numeric(1))
}

#' Year fraction under a day count convention
#'
#' The denominators come from the table `basis`: `act360` 360, `act365`
#' 365, `thirty360` 360 and `thirtyE360` 360. `actAct` splits the period by
#' calendar year instead.
#'
#' @param start,end `Date` vectors of equal length.
#' @param convention One of `"act360"`, `"act365"`, `"thirty360"`,
#'   `"thirtyE360"` or `"actAct"`.
#' @return A numeric vector of year fractions.
#' @export
yearFraction = function(start, end,
                        convention = c("act360", "act365", "thirty360", "thirtyE360",
                                       "actAct")) {
  convention = match.arg(convention)
  if (length(start) != length(end)) {
    stop("`start` and `end` must have the same length.", call. = FALSE)
  }
  basis = c(act360 = 360, act365 = 365, thirty360 = 360, thirtyE360 = 360)

  switch(convention,
    act360 = {
      days = dayCount(start, end, "actual")
      days / basis[["act360"]]
    },
    act365 = {
      days = dayCount(start, end, "actual")
      days / basis[["act365"]]
    },
    thirty360 = {
      days = dayCount(start, end, "thirty360")
      days / basis[["thirty360"]]
    },
    thirtyE360 = {
      days = dayCount(start, end, "thirtyE360")
      days / basis[["thirtyE360"]]
    },
    actAct = {
      actActFraction(start, end)
    }
  )
}

#' Compare day count conventions for a set of periods
#'
#' @param start,end `Date` vectors of equal length.
#' @param notional Notional amount used to show the accrued interest.
#' @param rate Annual interest rate as a decimal.
#' @param digits Number of decimal places kept in the fractions.
#' @return A data frame with columns `start`, `end`, `days`, and for each
#'   of `act360`, `act365`, `thirty360`, `thirtyE360` and `actAct` the year
#'   fraction (named after the convention) and the accrued interest (the
#'   convention name followed by `.interest`).
#' @export
dayCountTable = function(start, end, notional = 100, rate = 0.05, digits = 6L) {
  start = as.Date(start)
  end = as.Date(end)
  out = data.frame(start = start, end = end, days = dayCount(start, end, "actual"))

  # Convention `act360`
  fraction = yearFraction(start, end, "act360")
  out$act360 = round(fraction, digits)
  out$act360.interest = round(notional * rate * fraction, 2)
  # Convention `act365`
  fraction = yearFraction(start, end, "act365")
  out$act365 = round(fraction, digits)
  out$act365.interest = round(notional * rate * fraction, 2)
  # Convention `thirty360`
  fraction = yearFraction(start, end, "thirty360")
  out$thirty360 = round(fraction, digits)
  out$thirty360.interest = round(notional * rate * fraction, 2)
  # Convention `thirtyE360`
  fraction = yearFraction(start, end, "thirtyE360")
  out$thirtyE360 = round(fraction, digits)
  out$thirtyE360.interest = round(notional * rate * fraction, 2)
  # Convention `actAct`
  fraction = yearFraction(start, end, "actAct")
  out$actAct = round(fraction, digits)
  out$actAct.interest = round(notional * rate * fraction, 2)
  out
}

#' Shift dates by whole months, clamping to month end
#'
#' @param x A `Date` vector.
#' @param n Number of months to add.
#' @return A `Date` vector.
#' @noRd
addMonths = function(x, n) {
  lt = as.POSIXlt(as.Date(x))
  total = (lt$year + 1900L) * 12L + lt$mon + n
  first = as.Date(sprintf("%04d-%02d-01", total %/% 12L, total %% 12L + 1L))
  nextFirst = seq(first, by = "month", length.out = 2L)[2L]
  pmin(first + (lt$mday - 1L), nextFirst - 1L)
}

#' Accrual periods and interest between two dates
#'
#' Periods are generated forwards from `start`; the last period is a short
#' stub when `end` does not fall on a regular date. Months per period are
#' taken from `months`: `annual` 12, `semiannual` 6, `quarterly` 3 and
#' `monthly` 1.
#'
#' @param start,end Single `Date` values.
#' @param frequency One of `"annual"`, `"semiannual"`, `"quarterly"` or
#'   `"monthly"`.
#' @param convention Day count convention passed to [yearFraction()].
#' @param notional Notional amount on which interest accrues.
#' @param rate Annual interest rate as a decimal.
#' @return A data frame with columns `from`, `to`, `fraction` and
#'   `interest`.
#' @export
accrualPeriods = function(start, end,
                          frequency = c("annual", "semiannual", "quarterly", "monthly"),
                          convention = "act360", notional = 1, rate = 0) {
  frequency = match.arg(frequency)
  start = as.Date(start)
  end = as.Date(end)
  months = c(annual = 12L, semiannual = 6L, quarterly = 3L, monthly = 1L)
  step = months[[frequency]]

  boundaries = start
  k = 1L
  repeat {
    nxt = addMonths(start, k * step)
    if (nxt >= end) break
    boundaries = c(boundaries, nxt)
    k = k + 1L
  }
  boundaries = c(boundaries, end)

  from = boundaries[-length(boundaries)]
  to = boundaries[-1L]
  fraction = yearFraction(from, to, convention)
  data.frame(from = from, to = to, fraction = fraction,
    interest = notional * rate * fraction)
}
