isWeekend = function(x, weekend = c(6L, 7L)) {
  as.integer(format(as.Date(x), "%u")) %in% weekend
}
holidayRule = function(name, type, month = NULL, day = NULL) {
  structure(list(name = name, type = type, month = month, day = day), class = "holidayRule")
}
holidayTable = function(rules, years) {
  rows = lapply(rules, function(r) {
    data.frame(name = r$name, date = as.Date(sprintf("%04d-%02d-%02d", years, r$month, r$day)))
  })
  do.call(rbind, rows)
}

sat = as.Date("2027-12-25")
sun = as.Date("2022-12-25")
wed = as.Date("2024-12-25")
x = c(sat, sun, wed, NA)
stopifnot(identical(observedDate(x), x))
stopifnot(identical(observedDate(x, "nearest"), c(sat - 1, sun + 1, wed, NA)))
stopifnot(identical(observedDate(x, "nextMonday"), c(sat + 2, sun + 1, wed, NA)))
stopifnot(identical(observedDate(x, "previousFriday"), c(sat - 1, sun - 2, wed, NA)))
stopifnot(identical(observedDate(x, "sundayOnly"), c(sat, sun + 1, wed, NA)))

xmas = holidayRule("Christmas", "fixed", 12, 25)
ny = holidayRule("New Year", "fixed", 1, 1)
cal = makeCalendar("UK", list(xmas, ny))
stopifnot(inherits(cal, "workCalendar"), cal$observed == "nearest")
stopifnot(identical(names(cal), c("name", "rules", "observed", "weekend")))
stopifnot(length(makeCalendar("X", xmas)$rules) == 1L)
stopifnot(inherits(try(makeCalendar("X", list(1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(makeCalendar("X", observed = "bogus"), silent = TRUE), "try-error"))
stopifnot(inherits(try(makeCalendar("X", weekend = 8), silent = TRUE), "try-error"))
stopifnot(inherits(try(makeCalendar("X", weekend = c(6, 6)), silent = TRUE), "try-error"))
stopifnot(inherits(try(makeCalendar(""), silent = TRUE), "try-error"))
stopifnot(inherits(try(makeCalendar("X", list(xmas, xmas)), silent = TRUE), "try-error"))

h = calendarHolidays(cal, 2027)
stopifnot(identical(h, as.Date(c("2027-01-01", "2027-12-24"))))
stopifnot(length(calendarHolidays(makeCalendar("Empty"), 2024)) == 0L)

cal2 = makeCalendar("US", list(xmas, holidayRule("July 4", "fixed", 7, 4)), weekend = 7L)
mc = mergeCalendars(cal, cal2)
stopifnot(mc$name == "UK+US", length(mc$rules) == 3L, identical(mc$weekend, c(6L, 7L)))
stopifnot(mergeCalendars(cal, name = "Solo")$name == "Solo")
stopifnot(inherits(try(mergeCalendars(), silent = TRUE), "try-error"))

stopifnot(grepl("UK: 2 rule\\(s\\), weekend Sat/Sun", format(cal)))
out = capture.output(r <- print(cal))
stopifnot(identical(r, cal), length(out) == 3L, grepl("Christmas \\(fixed\\)", out[2]))

s = summary(cal, years = 2027)
stopifnot(inherits(s, "summary.workCalendar"), s$nHolidays == 2L, s$nWeekdayHolidays == 2L)
stopifnot(summary(makeCalendar("N", list(xmas), observed = "none"), 2027)$nWeekdayHolidays == 0L)
ps = capture.output(print(s))
stopifnot(length(ps) == 4L, grepl("2027-2027", ps[1]))
