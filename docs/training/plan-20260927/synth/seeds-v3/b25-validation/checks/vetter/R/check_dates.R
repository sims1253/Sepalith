msg <- function(expr) conditionMessage(tryCatch(expr, error = function(e) e))

d <- as.Date("2024-03-05")
stopifnot(identical(as_date_strict(d), d))
stopifnot(identical(as_date_strict(c("2024-03-05", "05.03.2024", "03/05/2024")), rep(d, 3)))
stopifnot(identical(as_date_strict(as.POSIXct("2024-03-05 10:00", tz = "UTC")), d))
stopifnot(identical(msg(as_date_strict(c("2024-03-05", "soon"), "when")),
                    "`when` has unparseable date(s): soon."))
stopifnot(identical(msg(as_date_strict(5, "when")),
                    "`when` must be a date or a string, not numeric."))

stopifnot(identical(check_date("2024-03-05"), d))
stopifnot(identical(msg(check_date(c("2024-01-01", "2024-01-02"), "s")),
                    "`s` must be a single date."))
stopifnot(identical(msg(check_date(NA_character_, "s")), "`s` must not be missing."))
stopifnot(identical(msg(check_date(d, "s", min = as.Date("2025-01-01"))),
                    "`s` must be on or after 2025-01-01."))
stopifnot(identical(msg(check_date(d, "s", max = as.Date("2020-01-01"))),
                    "`s` must be on or before 2020-01-01."))

r <- check_date_range("2024-01-01", "2024-01-31")
stopifnot(identical(names(r), c("from", "to", "days")), identical(r$days, 30L))
r2 <- check_date_range(NULL, NULL, default_from = d, default_to = d + 2)
stopifnot(identical(r2$from, d), identical(r2$to, d + 2))
stopifnot(identical(msg(check_date_range("2024-02-01", "2024-01-01")),
                    "`to` must not be before `from`."))
stopifnot(grepl("more than `max_days`", msg(check_date_range(d, d + 10, max_days = 5))))

stopifnot(identical(check_period("day", 2)$by, "2 days"))
stopifnot(identical(check_period("week")$approx_days, 7L))
stopifnot(identical(check_period("month", 2)$by, "2 months"))
stopifnot(identical(check_period("quarter")$by, "3 months"))
stopifnot(identical(check_period("year")$approx_days, 365L))
stopifnot(identical(msg(check_period("day", 0)), "`n` must be a positive whole number."))

ts <- check_timestamp("2024-03-05T10:20:30")
stopifnot(inherits(ts, "POSIXct"), identical(format(ts, "%H:%M"), "10:20"))
stopifnot(inherits(check_timestamp(Sys.time()), "POSIXct"))
stopifnot(grepl("unparseable timestamp", msg(check_timestamp("later", "t"))))
stopifnot(identical(msg(check_timestamp(1, "t")),
                    "`t` must be a timestamp or a string, not numeric."))
stopifnot(identical(msg(check_timestamp(as.POSIXct(NA), "t")), "`t` must not be missing."))

s <- check_schedule_args("2024-01-01", "2024-06-30", "month", 1, "UTC")
stopifnot(identical(names(s), c("from", "to", "period", "by", "tz")))
stopifnot(identical(s$by, "1 months"))
stopifnot(grepl("spans more than", msg(check_schedule_args(d, d + 3, "week", 1, "UTC"))))
stopifnot(identical(msg(check_schedule_args(d, d + 30, "day", 1, "Mars/Base")),
                    "`tz` must be a single known time zone name."))

stopifnot(identical(check_period("fortnight", 2)$by, "4 weeks"))
stopifnot(identical(check_period("fortnight")$approx_days, 14L))

ev <- check_event_dates("2024-05-02", "2024-05-04", "2024-05-01")
stopifnot(identical(names(ev), c("start", "end", "created", "updated", "duration")))
stopifnot(identical(ev$duration, 2L), identical(ev$updated, as.Date("2024-05-01")))
stopifnot(identical(check_event_dates(d, d, d, d + 9)$updated, d + 9))
stopifnot(identical(msg(check_event_dates(d, d - 1, d)), "`end` must not be before `start`."))
stopifnot(identical(msg(check_event_dates(d, d, d + 1)), "`start` must not be before `created`."))
stopifnot(identical(msg(check_event_dates(d, d, d, d - 1)),
                    "`updated` must not be before `created`."))
stopifnot(identical(msg(check_event_dates("x", d, d)), "`start` has unparseable date(s): x."))
stopifnot(is.na(check_date(NA_character_, allow_na = TRUE)))
