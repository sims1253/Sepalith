stopifnot(identical(easterSunday(c(2024, 2025, 2019, 2038)),
  as.Date(c("2024-03-31", "2025-04-20", "2019-04-21", "2038-04-25"))))

stopifnot(nthWeekday(2024, 11, 4, 4) == as.Date("2024-11-28"))
stopifnot(nthWeekday(2024, 1, 1, 1) == as.Date("2024-01-01"))
stopifnot(is.na(nthWeekday(2024, 2, 1, 5)))
stopifnot(identical(nthWeekday(c(2023, 2024), 9, 1, 1), as.Date(c("2023-09-04", "2024-09-02"))))

stopifnot(lastWeekday(2024, 5, 1) == as.Date("2024-05-27"))
stopifnot(lastWeekday(2024, 12, 2) == as.Date("2024-12-31"))
stopifnot(identical(lastWeekday(c(2023, 2024), 2, 4), as.Date(c("2023-02-23", "2024-02-29"))))

xmas = holidayRule("Christmas", "fixed", month = 12, day = 25)
thanks = holidayRule("Thanksgiving", "nth", month = 11, weekday = 4, n = 4)
memorial = holidayRule("Memorial Day", "last", month = 5, weekday = 1)
goodFri = holidayRule("Good Friday", "easter", offset = -2)
stopifnot(inherits(xmas, "holidayRule"), identical(names(thanks), c("name", "type", "month",
  "weekday", "n")))
stopifnot(identical(names(goodFri), c("name", "type", "offset")))
stopifnot(inherits(try(holidayRule("X", "nth", month = 1, weekday = 1), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(holidayRule("X", "fixed", month = 13, day = 1), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(holidayRule("X", "fixed", month = 1, day = 32), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(holidayRule("X", "last", month = 1, weekday = 8), silent = TRUE),
  "try-error"))
stopifnot(inherits(try(holidayRule("X", "nth", month = 1, weekday = 1, n = 6), silent = TRUE),
  "try-error"))

stopifnot(holidayDate(xmas, 2024) == as.Date("2024-12-25"))
stopifnot(holidayDate(thanks, 2025) == as.Date("2025-11-27"))
stopifnot(identical(holidayDate(memorial, 2024:2025), as.Date(c("2024-05-27", "2025-05-26"))))
stopifnot(holidayDate(goodFri, 2024) == as.Date("2024-03-29"))

tab = holidayTable(list(xmas, thanks, memorial, goodFri), 2024:2025)
stopifnot(nrow(tab) == 8L, identical(names(tab), c("name", "type", "year", "date")))
stopifnot(!is.unsorted(tab$date), tab$name[1] == "Good Friday")
stopifnot(nrow(holidayTable(xmas, 2020)) == 1L)
stopifnot(nrow(holidayTable(holidayRule("Fifth Mon", "nth", month = 2, weekday = 1, n = 5),
  2024)) == 0L)

out = capture.output(r <- print(thanks))
stopifnot(identical(r, thanks), grepl("occurrence 4 of Thu in Nov", out))
stopifnot(grepl("every Dec 25", capture.output(print(xmas))))
stopifnot(grepl("last Mon in May", capture.output(print(memorial))))
stopifnot(grepl("Easter Sunday -2 days", capture.output(print(goodFri))))
