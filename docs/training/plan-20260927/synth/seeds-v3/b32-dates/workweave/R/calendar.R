#' Move holidays that fall on a weekend to an observed date
#'
#' @param x A `Date` vector of holidays.
#' @param rule One of `"none"` (keep the date), `"nearest"` (Saturday to
#'   Friday, Sunday to Monday), `"nextMonday"` (both to Monday),
#'   `"previousFriday"` (both to Friday) or `"sundayOnly"` (only Sunday
#'   moves, to Monday).
#' @export
#' @return A `Date` vector the same length as `x`.
observedDate = function(x, rule = c("none", "nearest", "nextMonday", "previousFriday",
                                    "sundayOnly")) {
  rule = match.arg(rule)
  x = as.Date(x)
  wday = as.integer(format(x, "%u"))
  sat = !is.na(wday) & wday == 6L
  sun = !is.na(wday) & wday == 7L

  switch(rule,
    none = {
      x
    },
    nearest = {
      x[sat] = x[sat] - 1L
      x[sun] = x[sun] + 1L
      x
    },
    nextMonday = {
      x[sat] = x[sat] + 2L
      x[sun] = x[sun] + 1L
      x
    },
    previousFriday = {
      x[sat] = x[sat] - 1L
      x[sun] = x[sun] - 2L
      x
    },
    sundayOnly = {
      x[sun] = x[sun] + 1L
      x
    }
  )
}

#' Create a working calendar
#'
#' @param name Calendar name.
#' @param rules A list of `holidayRule` objects.
#' @param observed Weekend observance rule passed to [observedDate()].
#' @param weekend ISO weekdays treated as the weekend.
#' @export
#' @return An object of class `workCalendar`: a list with elements `name`,
#'   `rules`, `observed` and `weekend`.
makeCalendar = function(name, rules = list(), observed = "nearest", weekend = c(6L, 7L)) {
  if (inherits(rules, "holidayRule")) {
    rules = list(rules)
  }
  rules = unname(rules)

  # Check `name`
  if (!is.character(name) || length(name) != 1L || is.na(name) || !nzchar(name)) {
    stop("`name` must be a single non-empty string.", call. = FALSE)
  }
  # Check `rules`
  ok = vapply(rules, inherits, logical(1), what = "holidayRule")
  if (!all(ok)) {
    stop("`rules` must be a list of `holidayRule` objects.", call. = FALSE)
  }
  ruleNames = vapply(rules, function(rule) rule$name, character(1))
  if (anyDuplicated(ruleNames) > 0L) {
    stop("`rules` must have unique names.", call. = FALSE)
  }
  # Check `observed`
  observed = match.arg(observed, eval(formals(observedDate)$rule))
  # Check `weekend`
  if (!all(weekend %in% 1:7) || anyDuplicated(weekend) > 0L) {
    stop("`weekend` must contain distinct ISO weekdays between 1 and 7.", call. = FALSE)
  }

  structure(
    list(name = name, rules = rules, observed = observed, weekend = as.integer(weekend)),
    class = "workCalendar"
  )
}

#' Holiday dates of a calendar
#'
#' @param cal A `workCalendar`.
#' @param years Integer vector of years.
#' @export
#' @return A sorted `Date` vector of observed holidays without duplicates.
calendarHolidays = function(cal, years) {
  if (length(cal$rules) == 0L) {
    return(as.Date(character()))
  }
  tab = holidayTable(cal$rules, years)
  sort(unique(observedDate(tab$date, cal$observed)))
}

#' Combine several calendars into one
#'
#' The combined calendar has the union of the rules, the observance rule of
#' the first calendar, and the union of the weekends.
#'
#' @param ... `workCalendar` objects.
#' @param name Name of the combined calendar; by default the input names
#'   joined with `"+"`.
#' @export
#' @return A `workCalendar`.
mergeCalendars = function(..., name = NULL) {
  cals = list(...)
  if (length(cals) == 0L) {
    stop("At least one calendar is needed.", call. = FALSE)
  }
  calNames = vapply(cals, function(cal) cal$name, character(1))
  rules = do.call(c, lapply(cals, function(cal) cal$rules))
  ruleNames = vapply(rules, function(rule) rule$name, character(1))
  rules = rules[!duplicated(ruleNames)]
  weekend = sort(unique(unlist(lapply(cals, function(cal) cal$weekend))))
  if (is.null(name)) {
    name = paste(calNames, collapse = "+")
  }
  makeCalendar(name, rules, observed = cals[[1L]]$observed, weekend = weekend)
}

#' Methods for working calendars
#'
#' @param x,object A `workCalendar`.
#' @param years Years whose holidays are listed.
#' @param ... Ignored.
#' @name workCalendarMethods
NULL

#' @rdname workCalendarMethods
#' @export
format.workCalendar = function(x, ...) {
  days = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
  sprintf("<workCalendar> %s: %d rule(s), weekend %s, observed '%s'",
    x$name, length(x$rules), paste(days[x$weekend], collapse = "/"), x$observed)
}

#' @rdname workCalendarMethods
#' @export
print.workCalendar = function(x, ...) {
  cat(format(x), "\n", sep = "")
  for (rule in x$rules) {
    cat("  - ", rule$name, " (", rule$type, ")\n", sep = "")
  }
  invisible(x)
}

#' @rdname workCalendarMethods
#' @export
summary.workCalendar = function(object, years = as.integer(format(Sys.Date(), "%Y")), ...) {
  dates = calendarHolidays(object, years)
  weekdayDates = dates[!isWeekend(dates, object$weekend)]
  structure(
    list(
      name = object$name,
      years = years,
      nRules = length(object$rules),
      nHolidays = length(dates),
      nWeekdayHolidays = length(weekdayDates),
      dates = dates
    ),
    class = "summary.workCalendar"
  )
}

#' @rdname workCalendarMethods
#' @export
print.summary.workCalendar = function(x, ...) {
  cat("Calendar ", x$name, ", ", min(x$years), "-", max(x$years), "\n", sep = "")
  cat("  rules:            ", x$nRules, "\n", sep = "")
  cat("  holidays:         ", x$nHolidays, "\n", sep = "")
  cat("  weekday holidays: ", x$nWeekdayHolidays, "\n", sep = "")
  invisible(x)
}
