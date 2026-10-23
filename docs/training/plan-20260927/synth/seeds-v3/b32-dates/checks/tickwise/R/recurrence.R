r <- rrule.parse("FREQ=WEEKLY;INTERVAL=2;COUNT=4;BYDAY=MO,TH")
stopifnot(identical(names(r), c("freq", "interval", "count", "until", "byday")))
stopifnot(r$freq == "weekly", r$interval == 2L, r$count == 4L, is.null(r$until))
stopifnot(identical(r$byday, c(1L, 4L)))
r2 <- rrule.parse("FREQ=MONTHLY;UNTIL=20241231")
stopifnot(r2$interval == 1L, r2$until == as.Date("2024-12-31"), is.null(r2$count))
stopifnot(inherits(try(rrule.parse("FREQ=DAILY;FOO=1"), silent = TRUE), "try-error"))
stopifnot(inherits(try(rrule.parse("FREQ=DAILY;COUNT"), silent = TRUE), "try-error"))

ok <- list(freq = "daily", interval = 1L, count = NULL, until = NULL, byday = NULL)
stopifnot(identical(rrule.validate(ok), ok))
bad <- function(...) inherits(try(rrule.validate(modifyList(ok, list(...))), silent = TRUE),
  "try-error")
stopifnot(bad(freq = "hourly"), bad(interval = 0L), bad(count = 0L))
stopifnot(bad(until = as.Date(NA)), bad(byday = 1L))

stopifnot(rrule.format(r) == "FREQ=WEEKLY;INTERVAL=2;COUNT=4;BYDAY=MO,TH")
stopifnot(rrule.format(r2) == "FREQ=MONTHLY;UNTIL=20241231")
stopifnot(identical(rrule.parse(rrule.format(r)), r))

s <- as.Date("2024-01-31")
stopifnot(rrule.step(s, "daily", 1L) == as.Date("2024-02-01"))
stopifnot(rrule.step(s, "weekly", 2L) == as.Date("2024-02-14"))
stopifnot(identical(rrule.step(s, "monthly", 0:2), as.Date(c("2024-01-31", NA, "2024-03-31"))))
stopifnot(identical(rrule.step(as.Date("2024-02-29"), "yearly", 0:1),
  as.Date(c("2024-02-29", NA))))

e <- rrule.expand(r, "2024-01-04")
stopifnot(identical(e, as.Date(c("2024-01-04", "2024-01-15", "2024-01-18", "2024-01-29"))))
m <- rrule.expand("FREQ=MONTHLY;COUNT=3", "2024-01-31")
stopifnot(identical(m, as.Date(c("2024-01-31", "2024-03-31", "2024-05-31"))))
stopifnot(length(rrule.expand("FREQ=DAILY;UNTIL=20240110", "2024-01-01")) == 10L)
stopifnot(length(rrule.expand("FREQ=YEARLY", "2024-01-01", limit = 5L)) == 5L)
