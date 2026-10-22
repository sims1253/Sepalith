#' Parse dates written in several formats
#'
#' Tries each format in turn and keeps the first one that parses each
#' element.
#'
#' @param x A character vector of dates.
#' @param formats Candidate formats passed to [as.Date()].
#' @return A `Date` vector the same length as `x`.
#' @export
parseDates <- function(x, formats = c("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y")) {
  out <- as.Date(rep(NA_character_, length(x)))
  for (fmt in formats) {
    todo <- is.na(out) & !is.na(x)
    if (!any(todo)) break
    out[todo] <- as.Date(x[todo], format = fmt)
  }
  failed <- sum(is.na(out) & !is.na(x))
  if (failed > 0) {
    warning(sprintf("%d value(s) matched none of the formats", failed), call. = FALSE)
  }
  out
}

#' Whole days between two dates
#'
#' @param from,to `Date` vectors, recycled to a common length.
#' @return A numeric vector of day counts; negative when `to` is earlier.
#' @family date differences
#' @export
daysBetween <- function(from, to) {
  as.numeric(difftime(as.Date(to), as.Date(from), units = "days"))
}

#' Whole weeks between two dates
#'
#' @param from,to `Date` vectors, recycled to a common length.
#' @return A numeric vector of complete weeks; negative when `to` is earlier.
#' @family date differences
#' @export
weeksBetween <- function(from, to) {
  trunc(as.numeric(difftime(as.Date(to), as.Date(from), units = "weeks")))
}

#' Whole months between two dates
#'
#' Counts calendar months, subtracting one when the day of the month in
#' `to` has not yet reached the day in `from`.
#'
#' @param from,to `Date` vectors, recycled to a common length.
#' @return An integer vector of complete months.
#' @family date differences
#' @export
monthsBetween <- function(from, to) {
  a <- as.POSIXlt(as.Date(from))
  b <- as.POSIXlt(as.Date(to))
  months <- (b$year - a$year) * 12L + (b$mon - a$mon)
  months - as.integer(b$mday < a$mday & months > 0)
}

#' First day of every month in a range
#'
#' @param from,to Dates bounding the range.
#' @return A `Date` vector of month starts from the month of `from` to the
#'   month of `to`.
#' @export
monthStarts <- function(from, to) {
  start <- as.Date(format(as.Date(from), "%Y-%m-01"))
  end <- as.Date(format(as.Date(to), "%Y-%m-01"))
  if (end < start) stop("'to' must not be before 'from'", call. = FALSE)
  seq(start, end, by = "month")
}

#' Working days in a range
#'
#' Lists Monday to Friday dates between two dates, inclusive, leaving out
#' any listed holidays.
#'
#' @param from,to Dates bounding the range.
#' @param holidays Optional `Date` vector of days to skip.
#' @return A `Date` vector.
#' @export
workingDays <- function(from, to, holidays = as.Date(character())) {
  days <- seq(as.Date(from), as.Date(to), by = "day")
  wday <- as.POSIXlt(days)$wday
  days[wday %in% 1:5 & !(days %in% as.Date(holidays))]
}

#' Label dates by period
#'
#' @param date A `Date` vector.
#' @param period One of "month", "quarter" or "year".
#' @return A character vector of labels such as `2024-03`, `2024-Q1` or
#'   `2024`.
#' @export
periodLabel <- function(date, period = c("month", "quarter", "year")) {
  period <- match.arg(period)
  date <- as.Date(date)
  switch(period,
    month = format(date, "%Y-%m"),
    quarter = sprintf("%s-Q%d", format(date, "%Y"),
                      (as.POSIXlt(date)$mon %/% 3L) + 1L),
    year = format(date, "%Y")
  )
}

#' Shift dates by whole months
#'
#' Days that do not exist in the target month are moved to its last day, so
#' January 31 plus one month gives the end of February.
#'
#' @param date A `Date` vector.
#' @param n Number of months to add; may be negative.
#' @return A `Date` vector the same length as `date`.
#' @export
addMonths <- function(date, n) {
  lt <- as.POSIXlt(as.Date(date))
  total <- lt$year * 12L + lt$mon + n
  year <- total %/% 12L + 1900L
  month <- total %% 12L + 1L
  first <- as.Date(sprintf("%04d-%02d-01", year, month))
  nextFirst <- as.Date(sprintf("%04d-%02d-01", (total + 1L) %/% 12L + 1900L,
                               (total + 1L) %% 12L + 1L))
  lastDay <- as.integer(format(nextFirst - 1, "%d"))
  first + pmin(lt$mday, lastDay) - 1L
}
