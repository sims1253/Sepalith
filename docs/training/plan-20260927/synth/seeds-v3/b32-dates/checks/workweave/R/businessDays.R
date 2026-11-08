fri = as.Date("2024-05-24")
hol = as.Date(c("2024-05-27", "2024-12-25"))

stopifnot(identical(isWeekend(fri + 0:3), c(FALSE, TRUE, TRUE, FALSE)))
stopifnot(identical(isWeekend(fri + 0:1, weekend = 5L), c(TRUE, FALSE)))

stopifnot(identical(isBusinessDay(fri + 0:4, hol), c(TRUE, FALSE, FALSE, FALSE, TRUE)))
stopifnot(isBusinessDay(fri + 3))

stopifnot(nextBusinessDay(fri, hol) == as.Date("2024-05-28"))
stopifnot(nextBusinessDay(fri) == as.Date("2024-05-27"))
stopifnot(previousBusinessDay(as.Date("2024-05-28"), hol) == fri)
stopifnot(identical(previousBusinessDay(as.Date(c("2024-05-27", NA))),
  as.Date(c("2024-05-24", NA))))

stopifnot(addBusinessDays(fri, 1, hol) == as.Date("2024-05-28"))
stopifnot(addBusinessDays(fri, 5, hol) == as.Date("2024-06-03"))
stopifnot(addBusinessDays(as.Date("2024-05-28"), -1, hol) == fri)
stopifnot(identical(addBusinessDays(fri + c(0, 1, 0), c(-2, 0, NA)),
  as.Date(c("2024-05-22", "2024-05-25", NA))))

stopifnot(businessDaysBetween(fri, fri + 3) == 1L)
stopifnot(businessDaysBetween(fri, fri + 3, hol) == 0L)
stopifnot(businessDaysBetween(fri + 7, fri, hol) == -4L)
stopifnot(identical(businessDaysBetween(c(fri, NA), c(fri, fri)), c(0L, NA)))

m = businessDaysInMonth(2024, 5, hol)
stopifnot(identical(names(m), c("days", "first", "last", "count", "holidays")))
stopifnot(m$count == 22L, m$first == as.Date("2024-05-01"), m$last == as.Date("2024-05-31"))
stopifnot(length(m$holidays) == 1L, m$holidays == as.Date("2024-05-27"))
stopifnot(businessDaysInMonth(2024, 12, hol)$count == 21L)

rep1 = businessDayReport("2024-05-24", "2024-05-31", hol)
stopifnot(identical(names(rep1), c("from", "to", "calendarDays", "weekendDays", "holidayDays",
  "businessDays", "firstBusinessDay", "lastBusinessDay", "share")))
stopifnot(rep1$calendarDays == 8L, rep1$weekendDays == 2L, rep1$holidayDays == 1L)
stopifnot(rep1$businessDays == 5L, rep1$firstBusinessDay == fri, rep1$share == 5 / 8)
rep2 = businessDayReport("2024-05-25", "2024-05-26")
stopifnot(rep2$businessDays == 0L, is.na(rep2$firstBusinessDay), is.na(rep2$lastBusinessDay))
stopifnot(inherits(try(businessDayReport(NA, fri), silent = TRUE), "try-error"))
stopifnot(inherits(try(businessDayReport(fri, NA), silent = TRUE), "try-error"))
stopifnot(inherits(try(businessDayReport(fri, fri - 1), silent = TRUE), "try-error"))
