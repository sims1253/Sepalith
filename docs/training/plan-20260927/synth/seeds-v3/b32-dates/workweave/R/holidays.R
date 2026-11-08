#' Date of Easter Sunday
#'
#' @description
#' Computus for the Gregorian calendar, vectorised over `year`.
#'
#' @param year Integer vector of years.
#' @return A `Date` vector.
#' @export
easterSunday = function(year) {
  year = as.integer(year)
  a = year %% 19L
  b = year %/% 100L
  c = year %% 100L
  d = (19L * a + b - b %/% 4L - (b - (b + 8L) %/% 25L + 1L) %/% 3L + 15L) %% 30L
  e = (32L + 2L * (b %% 4L) + 2L * (c %/% 4L) - d - c %% 4L) %% 7L
  f = (a + 11L * d + 22L * e) %/% 451L
  month = (d + e - 7L * f + 114L) %/% 31L
  day = (d + e - 7L * f + 114L) %% 31L + 1L
  as.Date(sprintf("%04d-%02d-%02d", year, month, day))
}

#' The n-th weekday of a month
#'
#' @param year Integer vector of years.
#' @param month Month number, 1 to 12.
#' @param weekday ISO weekday, 1 (Monday) to 7 (Sunday).
#' @param n Which occurrence, 1 to 5.
#' @export
#' @return A `Date` vector; `NA` where the month has no such occurrence.
nthWeekday = function(year, month, weekday, n) {
  first = as.Date(sprintf("%04d-%02d-01", as.integer(year), as.integer(month)))
  firstWday = as.integer(format(first, "%u"))
  offset = (weekday - firstWday) %% 7L
  out = first + offset + 7L * (n - 1L)
  out[format(out, "%m") != format(first, "%m")] = NA
  out
}

#' The last given weekday of a month
#'
#' @param year Integer vector of years.
#' @param month Month number, 1 to 12.
#' @param weekday ISO weekday, 1 (Monday) to 7 (Sunday).
#' @export
#' @return A `Date` vector.
lastWeekday = function(year, month, weekday) {
  # Index of the following month, counted from year 0
  total = as.integer(year) * 12L + as.integer(month)
  nextFirst = as.Date(sprintf("%04d-%02d-01", total %/% 12L, total %% 12L + 1L))
  last = nextFirst - 1L
  lastWday = as.integer(format(last, "%u"))
  last - (lastWday - weekday) %% 7L
}

#' Create a holiday rule
#'
#' @description
#' A rule describes how to find a holiday in any given year. `type` decides
#' which further fields are required:
#' * `"fixed"`: `month` and `day`;
#' * `"nth"`: `month`, `weekday` and `n`;
#' * `"last"`: `month` and `weekday`;
#' * `"easter"`: `offset` in days from Easter Sunday.
#'
#' @param name Holiday name.
#' @param type One of `"fixed"`, `"nth"`, `"last"` or `"easter"`.
#' @param month Month number, 1 to 12.
#' @param day Day of the month.
#' @param weekday ISO weekday, 1 (Monday) to 7 (Sunday).
#' @param n Which occurrence, 1 to 5.
#' @param offset Days relative to Easter Sunday.
#' @export
#' @return An object of class `holidayRule`.
holidayRule = function(name, type = c("fixed", "nth", "last", "easter"),
                       month = NULL, day = NULL, weekday = NULL, n = NULL,
                       offset = 0L) {
  type = match.arg(type)
  need = switch(type,
    fixed = c("month", "day"),
    nth = c("month", "weekday", "n"),
    last = c("month", "weekday"),
    easter = "offset"
  )
  given = list(month = month, day = day, weekday = weekday, n = n, offset = offset)
  missingFields = need[vapply(given[need], is.null, logical(1))]
  if (length(missingFields) > 0L) {
    stop(sprintf("A \"%s\" rule needs `%s`.", type, paste(missingFields, collapse = "`, `")),
      call. = FALSE)
  }

  # Validate `month`
  if (!is.null(month) && !(month %in% 1:12)) {
    stop("`month` must be a whole number between 1 and 12.", call. = FALSE)
  }
  # Validate `day`
  if (!is.null(day) && !(day %in% 1:31)) {
    stop("`day` must be a whole number between 1 and 31.", call. = FALSE)
  }
  # Validate `weekday`
  if (!is.null(weekday) && !(weekday %in% 1:7)) {
    stop("`weekday` must be a whole number between 1 and 7.", call. = FALSE)
  }
  # Validate `n`
  if (!is.null(n) && !(n %in% 1:5)) {
    stop("`n` must be a whole number between 1 and 5.", call. = FALSE)
  }
  structure(c(list(name = name, type = type), given[need]), class = "holidayRule")
}

#' Date of a holiday rule in given years
#'
#' @param rule A `holidayRule`.
#' @param years Integer vector of years.
#' @keywords internal
#' @return A `Date` vector the same length as `years`.
holidayDate = function(rule, years) {
  years = as.integer(years)
  switch(rule$type,
    fixed = {
      as.Date(sprintf("%04d-%02d-%02d", years, rule$month, rule$day))
    },
    nth = {
      nthWeekday(years, rule$month, rule$weekday, rule$n)
    },
    last = {
      lastWeekday(years, rule$month, rule$weekday)
    },
    easter = {
      easterSunday(years) + as.integer(rule$offset)
    }
  )
}

#' Expand holiday rules into a table of dates
#'
#' @param rules A list of `holidayRule` objects.
#' @param years Integer vector of years.
#' @export
#' @return A data frame with columns `name`, `type`, `year` and `date`,
#'   sorted by `date`, with rows whose date does not exist dropped.
holidayTable = function(rules, years) {
  if (inherits(rules, "holidayRule")) {
    rules = list(rules)
  }
  pieces = lapply(rules, function(rule) {
    data.frame(
      name = rule$name,
      type = rule$type,
      year = as.integer(years),
      date = holidayDate(rule, years)
    )
  })
  out = do.call(rbind, pieces)
  out = out[!is.na(out$date), , drop = FALSE]
  out = out[order(out$date), , drop = FALSE]
  rownames(out) = NULL
  out
}

#' Print a holiday rule
#'
#' @param x A `holidayRule`.
#' @param ... Ignored.
#' @export
#' @return `x`, invisibly.
print.holidayRule = function(x, ...) {
  days = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  desc = switch(x$type,
    fixed = sprintf("every %s %d", month.abb[x$month], x$day),
    nth = sprintf("occurrence %d of %s in %s", x$n, days[x$weekday], month.abb[x$month]),
    last = sprintf("last %s in %s", days[x$weekday], month.abb[x$month]),
    easter = sprintf("Easter Sunday %+d days", as.integer(x$offset))
  )
  cat("<holidayRule> ", x$name, ": ", desc, "\n", sep = "")
  invisible(x)
}
