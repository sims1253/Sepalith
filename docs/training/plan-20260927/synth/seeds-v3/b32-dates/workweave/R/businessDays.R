#' Test for weekend days
#'
#' @param x A `Date` vector.
#' @param weekend ISO weekdays treated as the weekend.
#' @keywords internal
#' @return A logical vector the same length as `x`.
isWeekend = function(x, weekend = c(6L, 7L)) {
  as.integer(format(as.Date(x), "%u")) %in% weekend
}

#' Test for business days
#'
#' @param x A `Date` vector.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A logical vector the same length as `x`.
isBusinessDay = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  x = as.Date(x)
  !isWeekend(x, weekend) & !(x %in% as.Date(holidays))
}

#' Move to the nearest business day
#'
#' `nextBusinessDay()` returns the first business day strictly after each
#' date; `previousBusinessDay()` the last one strictly before it.
#'
#' @param x A `Date` vector.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @rdname neighbourBusinessDay
#' @export
#' @return A `Date` vector the same length as `x`.
nextBusinessDay = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  out = as.Date(x) + 1L
  todo = !is.na(out) & !isBusinessDay(out, holidays, weekend)
  while (any(todo)) {
    out[todo] = out[todo] + 1L
    todo[todo] = !isBusinessDay(out[todo], holidays, weekend)
  }
  out
}

#' @rdname neighbourBusinessDay
#' @export
previousBusinessDay = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  out = as.Date(x) - 1L
  todo = !is.na(out) & !isBusinessDay(out, holidays, weekend)
  while (any(todo)) {
    out[todo] = out[todo] - 1L
    todo[todo] = !isBusinessDay(out[todo], holidays, weekend)
  }
  out
}

#' Add business days to dates
#'
#' Positive `n` steps forward with [nextBusinessDay()], negative `n` steps
#' backward with [previousBusinessDay()]. Adding zero days to a non-business
#' day returns the date unchanged.
#'
#' @param x A `Date` vector.
#' @param n Integer vector of business days to add (recycled).
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A `Date` vector the same length as `x`.
addBusinessDays = function(x, n, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  x = as.Date(x)
  n = rep_len(as.integer(n), length(x))
  out = x

  # Step forward for positive `n`
  fwd = which(!is.na(n) & n > 0L)
  left = n[fwd]
  while (length(fwd) > 0L) {
    out[fwd] = nextBusinessDay(out[fwd], holidays, weekend)
    left = left - 1L
    fwd = fwd[left > 0L]
    left = left[left > 0L]
  }
  # Step backward for negative `n`
  bwd = which(!is.na(n) & n < 0L)
  left = -n[bwd]
  while (length(bwd) > 0L) {
    out[bwd] = previousBusinessDay(out[bwd], holidays, weekend)
    left = left - 1L
    bwd = bwd[left > 0L]
    left = left[left > 0L]
  }
  out[is.na(n)] = NA
  out
}

#' Count business days between two dates
#'
#' Counts business days in the half-open interval `(from, to]`, so a
#' Friday to the following Monday is one business day. The count is
#' negative when `to` is before `from`.
#'
#' @param from,to `Date` vectors of the same length.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return An integer vector.
businessDaysBetween = function(from, to, holidays = as.Date(character()),
                               weekend = c(6L, 7L)) {
  from = as.Date(from)
  to = as.Date(to)
  out = integer(length(from))
  for (i in seq_along(from)) {
    if (is.na(from[i]) || is.na(to[i])) {
      out[i] = NA_integer_
    } else if (to[i] >= from[i]) {
      days = seq(from[i], to[i], by = "day")[-1L]
      out[i] = sum(isBusinessDay(days, holidays, weekend))
    } else {
      days = seq(to[i], from[i], by = "day")[-1L]
      out[i] = -sum(isBusinessDay(days, holidays, weekend))
    }
  }
  out
}

#' Business days of a month
#'
#' @param year Year as an integer.
#' @param month Month number, 1 to 12.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A list with elements `days` (all business days of the month),
#'   `first`, `last`, `count`, and `holidays` (holidays that fell on a
#'   weekday of the month).
businessDaysInMonth = function(year, month, holidays = as.Date(character()),
                               weekend = c(6L, 7L)) {
  first = as.Date(sprintf("%04d-%02d-01", as.integer(year), as.integer(month)))
  span = seq(first, by = "month", length.out = 2L)
  all = seq(span[1L], span[2L] - 1L, by = "day")
  holidays = as.Date(holidays)

  days = all[isBusinessDay(all, holidays, weekend)]
  hit = all[all %in% holidays & !isWeekend(all, weekend)]
  list(
    days = days,
    first = days[1L],
    last = days[length(days)],
    count = length(days),
    holidays = hit
  )
}

#' Business day breakdown of a date range
#'
#' @param from,to Single `Date` values bounding the range (inclusive).
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A list with elements `from`, `to`, `calendarDays`, `weekendDays`,
#'   `holidayDays` (holidays on a weekday), `businessDays`,
#'   `firstBusinessDay`, `lastBusinessDay` and `share` (business days as a
#'   fraction of calendar days).
businessDayReport = function(from, to, holidays = as.Date(character()),
                             weekend = c(6L, 7L)) {
  from = as.Date(from)
  to = as.Date(to)
  # Check `from`
  if (length(from) != 1L || is.na(from)) {
    stop("`from` must be a single non-missing date.", call. = FALSE)
  }
  # Check `to`
  if (length(to) != 1L || is.na(to)) {
    stop("`to` must be a single non-missing date.", call. = FALSE)
  }
  if (to < from) {
    stop("`to` must not be before `from`.", call. = FALSE)
  }

  days = seq(from, to, by = "day")
  weekendDay = isWeekend(days, weekend)
  holidayDay = !weekendDay & days %in% as.Date(holidays)
  business = days[!weekendDay & !holidayDay]
  list(
    from = from,
    to = to,
    calendarDays = length(days),
    weekendDays = sum(weekendDay),
    holidayDays = sum(holidayDay),
    businessDays = length(business),
    firstBusinessDay = if (length(business) > 0L) business[1L] else as.Date(NA),
    lastBusinessDay = if (length(business) > 0L) business[length(business)] else as.Date(NA),
    share = length(business) / length(days)
  )
}
