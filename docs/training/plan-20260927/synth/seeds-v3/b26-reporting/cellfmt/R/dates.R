#' Coerce input to a Date vector
#'
#' Accepts `Date`, `POSIXct`/`POSIXlt`, character and numeric input so that
#' every date formatter can share one entry point.
#'
#' @param x A vector of dates in one of the supported classes.
#' @param format Format used to parse character input.
#' @param origin Origin used to convert numeric input.
#' @return A `Date` vector the same length as `x`.
#' @keywords internal
as_report_date <- function(x, format = "%Y-%m-%d", origin = "1970-01-01") {
  if (inherits(x, "Date")) {
    return(x)
  }

  if (inherits(x, "POSIXt")) {
    return(as.Date(x))
  }

  if (is.character(x)) {
    out <- as.Date(x, format = format)
    bad <- !is.na(x) & is.na(out)
    if (any(bad)) {
      stop(sprintf("`x` has %d value(s) not matching `format`.", sum(bad)), call. = FALSE)
    }
    return(out)
  }

  if (is.numeric(x)) {
    return(as.Date(x, origin = origin))
  }

  stop("`x` must be a Date, POSIXct, character or numeric vector.", call. = FALSE)
}

#' First day of a month
#'
#' Month numbers outside 1-12 roll over into neighbouring years, so
#' `month_start(2024, 13)` is the first of January 2025.
#'
#' @param year Integer vector of years.
#' @param month Integer vector of month numbers.
#' @return A `Date` vector.
#' @keywords internal
month_start <- function(year, month) {
  year <- year + (month - 1L) %/% 12L
  month <- (month - 1L) %% 12L + 1L
  return(as.Date(sprintf("%d-%02d-01", year, month), format = "%Y-%m-%d"))
}

#' Format dates for display
#'
#' Month names come from [month.name] and [month.abb] so the output does not
#' depend on the locale.
#'
#' @param x A vector of dates in one of the supported classes.
#' @param style One of `"iso"`, `"long"`, `"short"`, `"month"` or `"quarter"`.
#' @param na_string String used in place of missing values.
#' @return A character vector the same length as `x`.
#' @family date formatters
#' @export
fmt_date <- function(x, style = c("iso", "long", "short", "month", "quarter"),
                     na_string = "") {
  style <- match.arg(style)
  x <- as_report_date(x)
  lt <- as.POSIXlt(x)

  out <- switch(style,
    iso = {
      format(x, "%Y-%m-%d")
    },
    long = {
      day <- lt$mday
      month <- month.name[lt$mon + 1L]
      year <- lt$year + 1900L
      sprintf("%d %s %d", day, month, year)
    },
    short = {
      day <- lt$mday
      month <- month.abb[lt$mon + 1L]
      year <- lt$year + 1900L
      sprintf("%02d %s %d", day, month, year)
    },
    month = {
      month <- month.name[lt$mon + 1L]
      year <- lt$year + 1900L
      sprintf("%s %d", month, year)
    },
    quarter = {
      quarter <- lt$mon %/% 3L + 1L
      year <- lt$year + 1900L
      sprintf("Q%d %d", quarter, year)
    }
  )
  out[is.na(x)] <- na_string
  return(out)
}

#' Format pairs of dates as ranges
#'
#' Reversed pairs are swapped, missing ends are shown as `open_label`, and a
#' range whose ends coincide is written as a single date.
#'
#' @param start,end Vectors of dates in one of the supported classes.
#' @param sep Separator placed between the two ends.
#' @param open_label Label used for a missing end.
#' @return A character vector the same length as `start`.
#' @family date formatters
#' @export
fmt_date_range <- function(start, end, sep = " to ", open_label = "open") {
  start <- as_report_date(start)
  end <- as_report_date(end)
  if (length(start) != length(end)) {
    stop("`start` and `end` must have the same length.", call. = FALSE)
  }

  swapped <- !is.na(start) & !is.na(end) & start > end
  if (any(swapped)) {
    first <- end[swapped]
    end[swapped] <- start[swapped]
    start[swapped] <- first
  }

  # Format `start`
  left <- fmt_date(start, "iso", na_string = open_label)
  # Format `end`
  right <- fmt_date(end, "iso", na_string = open_label)
  out <- paste0(left, sep, right)
  same <- !is.na(start) & !is.na(end) & start == end
  out[same] <- left[same]
  return(out)
}

#' Split dates into calendar parts
#'
#' @param x A vector of dates in one of the supported classes.
#' @return A data frame with columns `year`, `quarter`, `month`,
#'   `month_name`, `day`, `day_of_year`, `week`, `weekday` and `weekend`, one
#'   row per element of `x`; `week` is the ISO 8601 week number.
#' @export
date_parts <- function(x) {
  x <- as_report_date(x)
  lt <- as.POSIXlt(x)
  weekday_names <- c(
    "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"
  )

  # Calendar position
  year <- lt$year + 1900L
  quarter <- lt$mon %/% 3L + 1L
  month <- lt$mon + 1L
  month_name <- month.name[month]
  day <- lt$mday
  day_of_year <- lt$yday + 1L

  # Day of the week
  week <- as.integer(format(x, "%V"))
  weekday <- weekday_names[lt$wday + 1L]
  weekend <- lt$wday %in% c(0L, 6L)
  weekend[is.na(x)] <- NA

  return(data.frame(
    year = year,
    quarter = quarter,
    month = month,
    month_name = month_name,
    day = day,
    day_of_year = day_of_year,
    week = week,
    weekday = weekday,
    weekend = weekend
  ))
}

#' Start and end of the enclosing period
#'
#' Weeks start on Monday; halves run from January and July.
#'
#' @param x A vector of dates in one of the supported classes.
#' @param period One of `"day"`, `"week"`, `"month"`, `"quarter"`, `"half"`
#'   or `"year"`.
#' @return A data frame with columns `start`, `end` and `days`.
#' @export
period_bounds <- function(x, period = c("day", "week", "month", "quarter", "half", "year")) {
  period <- match.arg(period)
  x <- as_report_date(x)
  lt <- as.POSIXlt(x)
  year <- lt$year + 1900L

  switch(period,
    day = {
      start <- x
      end <- x
    },
    week = {
      start <- x - (lt$wday + 6L) %% 7L
      end <- start + 6L
    },
    month = {
      start <- month_start(year, lt$mon + 1L)
      end <- month_start(year, lt$mon + 2L) - 1L
    },
    quarter = {
      first <- (lt$mon %/% 3L) * 3L
      start <- month_start(year, first + 1L)
      end <- month_start(year, first + 4L) - 1L
    },
    half = {
      first <- (lt$mon %/% 6L) * 6L
      start <- month_start(year, first + 1L)
      end <- month_start(year, first + 7L) - 1L
    },
    year = {
      start <- month_start(year, 1L)
      end <- month_start(year, 13L) - 1L
    }
  )
  days <- as.integer(end - start) + 1L
  return(data.frame(start = start, end = end, days = days))
}

#' Format elapsed time
#'
#' With `units = "auto"` each value uses the largest unit it reaches.
#'
#' @param seconds A numeric vector of durations in seconds.
#' @param units One of `"auto"`, `"secs"`, `"mins"`, `"hours"` or `"days"`.
#' @param digits Number of decimal places to keep.
#' @return A character vector the same length as `seconds`.
#' @family date formatters
#' @export
fmt_duration <- function(seconds, units = c("auto", "secs", "mins", "hours", "days"),
                         digits = 1) {
  units <- match.arg(units)
  sizes <- c(secs = 1, mins = 60, hours = 3600, days = 86400)
  if (units == "auto") {
    idx <- findInterval(abs(seconds), sizes)
    idx[!is.na(idx) & idx < 1L] <- 1L
    chosen <- names(sizes)[idx]
  } else {
    chosen <- rep(units, length(seconds))
  }

  value <- seconds / sizes[chosen]
  out <- paste(formatC(value, format = "f", digits = digits), chosen)
  out[is.na(seconds)] <- NA_character_
  return(out)
}

#' Describe dates relative to a reference day
#'
#' @param x A vector of dates in one of the supported classes.
#' @param reference A single reference date.
#' @return A character vector such as `"3 days ago"`, `"today"` or
#'   `"in 1 day"`.
#' @family date formatters
#' @export
fmt_relative <- function(x, reference = Sys.Date()) {
  x <- as_report_date(x)
  reference <- as_report_date(reference)
  gap <- as.integer(x - reference)
  out <- rep("today", length(gap))
  past <- !is.na(gap) & gap < 0L
  future <- !is.na(gap) & gap > 0L
  # Dates before `reference`
  n_past <- abs(gap[past])
  out[past] <- sprintf("%d %s ago", n_past, ifelse(n_past == 1L, "day", "days"))
  # Dates after `reference`
  n_future <- gap[future]
  out[future] <- sprintf("in %d %s", n_future, ifelse(n_future == 1L, "day", "days"))
  out[is.na(gap)] <- NA_character_
  return(out)
}
