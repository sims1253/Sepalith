#' Shift dates by whole months, clamping to the target day
#'
#' @param x A `Date` vector.
#' @param k Integer vector of months to add (recycled).
#' @param day Day of month to aim for; clamped to the month length.
#' @return A `Date` vector.
#' @keywords internal
#' @noRd
months.add <- function(x, k, day) {
  idx <- as.integer(format(x, "%Y")) * 12L + as.integer(format(x, "%m")) - 1L + k
  first <- as.Date(sprintf("%04d-%02d-01", idx %/% 12L, idx %% 12L + 1L))
  nxt <- as.Date(sprintf("%04d-%02d-01", (idx + 1L) %/% 12L, (idx + 1L) %% 12L + 1L))
  len <- as.integer(nxt - first)
  first + pmin(day, len) - 1L
}

#' @title Daily schedule
#' @param start,end Single `Date` values bounding the schedule (inclusive).
#' @param step Number of days between consecutive dates.
#' @return A `Date` vector.
#' @family schedule generators
#' @export
sched.daily <- function(start, end, step = 1L) {
  start <- as.Date(start)
  end <- as.Date(end)
  if (end < start) {
    return(as.Date(character()))
  }
  seq(start, end, by = step)
}

#' @title Weekly schedule
#' @param start,end Single `Date` values bounding the schedule (inclusive).
#' @param weekday ISO weekday (1 = Monday) to anchor on; `NULL` uses the
#'   weekday of `start`.
#' @return A `Date` vector.
#' @family schedule generators
#' @export
sched.weekly <- function(start, end, weekday = NULL) {
  start <- as.Date(start)
  end <- as.Date(end)
  if (!is.null(weekday)) {
    start <- start + (weekday - as.integer(format(start, "%u"))) %% 7L
  }
  if (end < start) {
    return(as.Date(character()))
  }
  seq(start, end, by = 7L)
}

#' @title Monthly schedule
#' @param start,end Single `Date` values bounding the schedule (inclusive).
#' @param day Day of month to anchor on; `NULL` uses the day of `start`.
#'   Months that are too short use their last day.
#' @return A `Date` vector.
#' @family schedule generators
#' @export
sched.monthly <- function(start, end, day = NULL) {
  start <- as.Date(start)
  end <- as.Date(end)
  if (is.null(day)) {
    day <- as.integer(format(start, "%d"))
  }
  n <- (as.integer(format(end, "%Y")) - as.integer(format(start, "%Y"))) * 12L +
    as.integer(format(end, "%m")) - as.integer(format(start, "%m"))
  out <- months.add(start, 0:max(n, 0L), day)
  out[out >= start & out <= end]
}

#' @title Quarterly schedule
#' @param start,end Single `Date` values bounding the schedule (inclusive).
#' @param day Day of month to anchor on; `NULL` uses the day of `start`.
#'   Months that are too short use their last day.
#' @return A `Date` vector.
#' @family schedule generators
#' @export
sched.quarterly <- function(start, end, day = NULL) {
  start <- as.Date(start)
  end <- as.Date(end)
  if (is.null(day)) {
    day <- as.integer(format(start, "%d"))
  }
  n <- (as.integer(format(end, "%Y")) - as.integer(format(start, "%Y"))) * 12L +
    as.integer(format(end, "%m")) - as.integer(format(start, "%m"))
  out <- months.add(start, seq(0L, max(n, 0L), by = 3L), day)
  out[out >= start & out <= end]
}

#' @title Yearly schedule
#' @param start,end Single `Date` values bounding the schedule (inclusive).
#' @param day Day of month to anchor on; `NULL` uses the day of `start`.
#'   Months that are too short use their last day.
#' @return A `Date` vector.
#' @family schedule generators
#' @export
sched.yearly <- function(start, end, day = NULL) {
  start <- as.Date(start)
  end <- as.Date(end)
  if (is.null(day)) {
    day <- as.integer(format(start, "%d"))
  }
  n <- (as.integer(format(end, "%Y")) - as.integer(format(start, "%Y"))) * 12L +
    as.integer(format(end, "%m")) - as.integer(format(start, "%m"))
  out <- months.add(start, seq(0L, max(n, 0L), by = 12L), day)
  out[out >= start & out <= end]
}

#' @title Build a schedule for a frequency
#' @param start,end Single `Date` values bounding the schedule (inclusive).
#' @param freq One of `"daily"`, `"weekly"`, `"monthly"`, `"quarterly"` or
#'   `"yearly"`.
#' @param anchor Optional anchor passed on as `weekday` for weekly
#'   schedules and as `day` for monthly, quarterly and yearly ones.
#' @param max.n Upper limit on the number of dates; longer schedules are an
#'   error.
#' @return A `Date` vector of class `c("tw_schedule", "Date")` with
#'   attribute `freq`.
#' @export
make.schedule <- function(start, end, freq = c("daily", "weekly", "monthly", "quarterly",
                                               "yearly"),
                          anchor = NULL, max.n = 10000L) {
  freq <- match.arg(freq)
  start <- as.Date(start)
  end <- as.Date(end)

  # Check `start`
  if (length(start) != 1L || is.na(start)) {
    stop("`start` must be a single non-missing date.", call. = FALSE)
  }
  # Check `end`
  if (length(end) != 1L || is.na(end)) {
    stop("`end` must be a single non-missing date.", call. = FALSE)
  }
  # Check `anchor`
  if (!is.null(anchor)) {
    limit <- if (freq == "weekly") 7L else 31L
    if (freq == "daily" || length(anchor) != 1L || !(anchor %in% seq_len(limit))) {
      stop(sprintf("`anchor` must be NULL or a whole number from 1 to %d for \"%s\".",
        limit, freq), call. = FALSE)
    }
  }

  out <- switch(freq,
    daily = sched.daily(start, end),
    weekly = sched.weekly(start, end, weekday = anchor),
    monthly = sched.monthly(start, end, day = anchor),
    quarterly = sched.quarterly(start, end, day = anchor),
    yearly = sched.yearly(start, end, day = anchor)
  )
  if (length(out) > max.n) {
    stop(sprintf("Schedule has %d dates, more than `max.n` (%d).", length(out), max.n),
      call. = FALSE)
  }
  structure(out, freq = freq, class = c("tw_schedule", "Date"))
}

#' @title Accrual periods between schedule dates
#' @param dates A sorted `Date` vector, e.g. from [make.schedule()].
#' @param day.count One of `"act360"`, `"act365"`, `"actact"`, `"thirty360"`
#'   or `"thirtyE360"`.
#' @return A data frame with one row per consecutive pair of dates and
#'   columns `start`, `end`, `days` and `fraction` (year fraction under
#'   `day.count`).
#' @keywords internal
schedule.periods <- function(dates, day.count = c("act360", "act365", "actact", "thirty360",
                                                  "thirtyE360")) {
  day.count <- match.arg(day.count)
  dates <- sort(as.Date(unclass(dates)))
  if (length(dates) < 2L) {
    stop("At least two dates are needed to form a period.", call. = FALSE)
  }
  start <- dates[-length(dates)]
  end <- dates[-1L]
  days <- as.integer(end - start)

  fraction <- switch(day.count,
    act360 = days / 360,
    act365 = days / 365,
    actact = {
      year <- as.integer(format(start, "%Y"))
      leap <- (year %% 4L == 0L & year %% 100L != 0L) | year %% 400L == 0L
      days / ifelse(leap, 366, 365)
    },
    thirty360 = {
      d1 <- pmin(as.integer(format(start, "%d")), 30L)
      d2 <- ifelse(d1 == 30L, pmin(as.integer(format(end, "%d")), 30L),
        as.integer(format(end, "%d")))
      y <- as.integer(format(end, "%Y")) - as.integer(format(start, "%Y"))
      m <- as.integer(format(end, "%m")) - as.integer(format(start, "%m"))
      (360L * y + 30L * m + d2 - d1) / 360
    },
    thirtyE360 = {
      d1 <- pmin(as.integer(format(start, "%d")), 30L)
      d2 <- pmin(as.integer(format(end, "%d")), 30L)
      y <- as.integer(format(end, "%Y")) - as.integer(format(start, "%Y"))
      m <- as.integer(format(end, "%m")) - as.integer(format(start, "%m"))
      (360L * y + 30L * m + d2 - d1) / 360
    }
  )
  data.frame(start = start, end = end, days = days, fraction = fraction)
}
