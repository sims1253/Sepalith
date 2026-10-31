#' First day of the period containing each date
#'
#' @param x A `Date` vector.
#' @param unit One of `"week"`, `"month"`, `"quarter"`, `"half"` or `"year"`.
#' @param week_start Day the week starts on, 1 (Monday) to 7 (Sunday).
#' @return A `Date` vector the same length as `x`.
#' @family period boundaries
#' @export
start_of_period <- function(x, unit = c("week", "month", "quarter", "half", "year"),
                            week_start = 1L) {
  unit <- match.arg(unit)
  if (length(week_start) != 1L || !(week_start %in% 1:7)) {
    stop("`week_start` must be a single weekday number from 1 to 7.", call. = FALSE)
  }
  x <- as.Date(x)
  lt <- as.POSIXlt(x)
  year <- lt$year + 1900L
  mon <- lt$mon + 1L

  out <- switch(unit,
    week = {
      wday <- as.integer(format(x, "%u"))
      x - (wday - week_start) %% 7L
    },
    month = {
      as.Date(sprintf("%04d-%02d-01", year, mon))
    },
    quarter = {
      first_mon <- (mon - 1L) %/% 3L * 3L + 1L
      as.Date(sprintf("%04d-%02d-01", year, first_mon))
    },
    half = {
      first_mon <- (mon - 1L) %/% 6L * 6L + 1L
      as.Date(sprintf("%04d-%02d-01", year, first_mon))
    },
    year = {
      as.Date(sprintf("%04d-01-01", year))
    }
  )
  return(out)
}

#' Last day of the period containing each date
#'
#' @inheritParams start_of_period
#' @return A `Date` vector the same length as `x`.
#' @family period boundaries
#' @export
end_of_period <- function(x, unit = c("week", "month", "quarter", "half", "year"),
                          week_start = 1L) {
  unit <- match.arg(unit)
  if (length(week_start) != 1L || !(week_start %in% 1:7)) {
    stop("`week_start` must be a single weekday number from 1 to 7.", call. = FALSE)
  }
  x <- as.Date(x)
  lt <- as.POSIXlt(x)
  year <- lt$year + 1900L
  mon <- lt$mon + 1L

  out <- switch(unit,
    week = {
      wday <- as.integer(format(x, "%u"))
      x + (week_start - 1L - wday) %% 7L
    },
    month = {
      first <- as.Date(sprintf("%04d-%02d-01", year, mon))
      first + (days_in_month(first) - 1L)
    },
    quarter = {
      last_mon <- (mon - 1L) %/% 3L * 3L + 3L
      first <- as.Date(sprintf("%04d-%02d-01", year, last_mon))
      first + (days_in_month(first) - 1L)
    },
    half = {
      last_mon <- (mon - 1L) %/% 6L * 6L + 6L
      first <- as.Date(sprintf("%04d-%02d-01", year, last_mon))
      first + (days_in_month(first) - 1L)
    },
    year = {
      as.Date(sprintf("%04d-12-31", year))
    }
  )
  return(out)
}

#' Test whether dates fall on a period boundary
#'
#' @inheritParams start_of_period
#' @return A logical vector the same length as `x`.
#' @rdname period_boundary_tests
#' @export
is_period_start <- function(x, unit = c("week", "month", "quarter", "half", "year"),
                            week_start = 1L) {
  unit <- match.arg(unit)
  x <- as.Date(x)
  return(x == start_of_period(x, unit, week_start))
}

#' @rdname period_boundary_tests
#' @export
is_period_end <- function(x, unit = c("week", "month", "quarter", "half", "year"),
                          week_start = 1L) {
  unit <- match.arg(unit)
  x <- as.Date(x)
  return(x == end_of_period(x, unit, week_start))
}

#' Period bounds as a data frame
#'
#' @inheritParams start_of_period
#' @return A data frame with columns `date`, `start`, `end`, `length` (days
#'   in the period) and `elapsed` (days since `start`, starting at 1).
#' @family period boundaries
#' @export
period_bounds <- function(x, unit = c("week", "month", "quarter", "half", "year"),
                          week_start = 1L) {
  unit <- match.arg(unit)
  x <- as.Date(x)
  start <- start_of_period(x, unit, week_start)
  end <- end_of_period(x, unit, week_start)
  length <- as.integer(end - start) + 1L
  elapsed <- as.integer(x - start) + 1L
  return(data.frame(
    date = x,
    start = start,
    end = end,
    length = length,
    elapsed = elapsed
  ))
}

#' Clamp dates into a window
#'
#' Dates before `lower` are moved to `lower` and dates after `upper` are
#' moved to `upper`. Either bound may be `NULL` to leave that side open.
#'
#' @param x A `Date` vector.
#' @param lower,upper Single `Date` values or `NULL`.
#' @return A `Date` vector the same length as `x`.
#' @keywords internal
clamp_dates <- function(x, lower = NULL, upper = NULL) {
  x <- as.Date(x)
  if (!is.null(lower) && !is.null(upper) && as.Date(lower) > as.Date(upper)) {
    stop("`lower` must not be after `upper`.", call. = FALSE)
  }

  # Move early dates up to `lower`
  if (!is.null(lower)) {
    lower <- as.Date(lower)
    early <- !is.na(x) & x < lower
    x[early] <- lower
  }
  # Move late dates down to `upper`
  if (!is.null(upper)) {
    upper <- as.Date(upper)
    late <- !is.na(x) & x > upper
    x[late] <- upper
  }
  return(x)
}

#' Fraction of the period elapsed at each date
#'
#' @inheritParams start_of_period
#' @param inclusive If `TRUE`, the end day counts as fully elapsed.
#' @return A numeric vector of values in `[0, 1]`.
#' @keywords internal
period_progress <- function(x, unit = c("week", "month", "quarter", "half", "year"),
                            week_start = 1L, inclusive = TRUE) {
  unit <- match.arg(unit)
  b <- period_bounds(x, unit, week_start)
  done <- if (inclusive) b$elapsed else b$elapsed - 1L
  return(done / b$length)
}
