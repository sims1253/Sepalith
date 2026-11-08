#' Roll non-business days to a neighbouring business day
#'
#' `rollFollowing()` keeps business days and moves every other date to the
#' next business day; `rollPreceding()` moves it to the previous one.
#'
#' @param x A `Date` vector.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @rdname rollDays
#' @keywords internal
#' @return A `Date` vector the same length as `x`.
rollFollowing = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  x = as.Date(x)
  move = !is.na(x) & !isBusinessDay(x, holidays, weekend)
  x[move] = nextBusinessDay(x[move], holidays, weekend)
  x
}

#' @rdname rollDays
#' @keywords internal
rollPreceding = function(x, holidays = as.Date(character()), weekend = c(6L, 7L)) {
  x = as.Date(x)
  move = !is.na(x) & !isBusinessDay(x, holidays, weekend)
  x[move] = previousBusinessDay(x[move], holidays, weekend)
  x
}

#' Adjust dates with a business day convention
#'
#' @param x A `Date` vector.
#' @param convention One of `"unadjusted"`, `"following"`, `"preceding"`,
#'   `"modifiedFollowing"`, `"modifiedPreceding"` or `"nearest"`. The
#'   modified conventions fall back to the opposite direction when rolling
#'   would cross into another month; `"nearest"` takes whichever business
#'   day is closer, preferring the following one on ties.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A `Date` vector the same length as `x`.
adjustDate = function(x, convention = c("unadjusted", "following", "preceding",
                                        "modifiedFollowing", "modifiedPreceding",
                                        "nearest"),
                      holidays = as.Date(character()), weekend = c(6L, 7L)) {
  convention = match.arg(convention)
  x = as.Date(x)
  month = format(x, "%Y-%m")

  switch(convention,
    unadjusted = {
      x
    },
    following = {
      rollFollowing(x, holidays, weekend)
    },
    preceding = {
      rollPreceding(x, holidays, weekend)
    },
    modifiedFollowing = {
      out = rollFollowing(x, holidays, weekend)
      crossed = !is.na(x) & format(out, "%Y-%m") != month
      out[crossed] = rollPreceding(x[crossed], holidays, weekend)
      out
    },
    modifiedPreceding = {
      out = rollPreceding(x, holidays, weekend)
      crossed = !is.na(x) & format(out, "%Y-%m") != month
      out[crossed] = rollFollowing(x[crossed], holidays, weekend)
      out
    },
    nearest = {
      out = rollFollowing(x, holidays, weekend)
      back = rollPreceding(x, holidays, weekend)
      closer = !is.na(x) & (x - back) < (out - x)
      out[closer] = back[closer]
      out
    }
  )
}

#' Adjust a schedule and report what moved
#'
#' @param dates A `Date` vector of unadjusted dates.
#' @param convention Business day convention passed to [adjustDate()].
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A data frame with columns `original`, `adjusted`, `shift` (signed
#'   calendar days moved) and `moved` (logical).
adjustSchedule = function(dates, convention = "modifiedFollowing",
                          holidays = as.Date(character()), weekend = c(6L, 7L)) {
  original = as.Date(dates)
  adjusted = adjustDate(original, convention, holidays, weekend)
  shift = as.integer(adjusted - original)
  moved = !is.na(shift) & shift != 0L
  data.frame(
    original = original,
    adjusted = adjusted,
    shift = shift,
    moved = moved
  )
}

#' First and last business days of each month
#'
#' @param x A `Date` vector; only the month of each date is used.
#' @param side Either `"first"` or `"last"`.
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A `Date` vector the same length as `x`.
monthBusinessDay = function(x, side = c("first", "last"),
                            holidays = as.Date(character()), weekend = c(6L, 7L)) {
  side = match.arg(side)
  x = as.Date(x)
  first = as.Date(format(x, "%Y-%m-01"))
  if (side == "first") {
    rollFollowing(first, holidays, weekend)
  } else {
    total = as.integer(format(first, "%Y")) * 12L + as.integer(format(first, "%m"))
    nextFirst = as.Date(sprintf("%04d-%02d-01", total %/% 12L, total %% 12L + 1L))
    rollPreceding(nextFirst - 1L, holidays, weekend)
  }
}

#' Compare business day conventions side by side
#'
#' @param x A `Date` vector.
#' @param conventions Character vector of conventions accepted by
#'   [adjustDate()].
#' @param holidays A `Date` vector of non-working days.
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return A data frame with a `date` column followed by one `Date` column
#'   per convention, named after it.
compareConventions = function(x, conventions = c("following", "preceding",
                                                 "modifiedFollowing",
                                                 "modifiedPreceding"),
                              holidays = as.Date(character()), weekend = c(6L, 7L)) {
  out = data.frame(date = as.Date(x))
  for (conv in conventions) {
    out[[conv]] = adjustDate(out$date, conv, holidays, weekend)
  }
  out
}
