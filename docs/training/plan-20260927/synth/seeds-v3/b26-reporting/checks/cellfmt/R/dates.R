d <- as.Date("2024-03-05")
stopifnot(identical(as_report_date(d), d))
stopifnot(identical(as_report_date("2024-03-05"), d))
stopifnot(identical(as_report_date(as.POSIXct("2024-03-05 10:00", tz = "UTC")), d))
stopifnot(identical(as_report_date(19787), d))
stopifnot(inherits(try(as_report_date("05/03/2024"), silent = TRUE), "try-error"))
stopifnot(inherits(try(as_report_date(TRUE), silent = TRUE), "try-error"))

stopifnot(identical(month_start(2024, 13), as.Date("2025-01-01")))
stopifnot(identical(month_start(2024, 3), as.Date("2024-03-01")))
stopifnot(identical(month_start(2024, 0), as.Date("2023-12-01")))

stopifnot(identical(fmt_date(d), "2024-03-05"))
stopifnot(identical(fmt_date(d, "long"), "5 March 2024"))
stopifnot(identical(fmt_date(d, "short"), "05 Mar 2024"))
stopifnot(identical(fmt_date(d, "month"), "March 2024"))
stopifnot(identical(fmt_date(c(d, NA), "quarter", na_string = "-"), c("Q1 2024", "-")))

r <- fmt_date_range(c("2024-01-01", "2024-02-10", "2024-05-01", NA),
                    c("2024-01-31", "2024-02-01", "2024-05-01", "2024-06-01"))
stopifnot(identical(r[1], "2024-01-01 to 2024-01-31"))
stopifnot(identical(r[2], "2024-02-01 to 2024-02-10"))
stopifnot(identical(r[3:4], c("2024-05-01", "open to 2024-06-01")))
stopifnot(inherits(try(fmt_date_range(d, c(d, d)), silent = TRUE), "try-error"))

p <- date_parts(c("2024-03-05", "2023-12-31"))
stopifnot(identical(names(p), c("year", "quarter", "month", "month_name", "day",
                               "day_of_year", "week", "weekday", "weekend")))
stopifnot(identical(p$weekday, c("Tuesday", "Sunday")), p$quarter == c(1L, 4L))
stopifnot(p$year == c(2024L, 2023L), p$day == c(5L, 31L))

stopifnot(identical(period_bounds(d, "week")$start, as.Date("2024-03-04")))
stopifnot(identical(period_bounds(d, "week")$end, as.Date("2024-03-10")))
stopifnot(identical(period_bounds(d, "month")$end, as.Date("2024-03-31")))
q <- period_bounds("2024-11-15", "quarter")
stopifnot(identical(q$start, as.Date("2024-10-01")), identical(q$end, as.Date("2024-12-31")))
stopifnot(identical(period_bounds(d, "year")$end, as.Date("2024-12-31")))

stopifnot(identical(fmt_duration(c(30, 90, 7200, 172800, NA)),
                    c("30.0 secs", "1.5 mins", "2.0 hours", "2.0 days", NA)))
stopifnot(identical(fmt_duration(0.5), "0.5 secs"))
stopifnot(identical(fmt_duration(120, "secs", digits = 0), "120 secs"))

rel <- fmt_relative(c("2024-03-02", "2024-03-05", "2024-03-06", NA), reference = d)
stopifnot(identical(rel, c("3 days ago", "today", "in 1 day", NA)))
stopifnot(identical(fmt_relative("2024-03-04", d), "1 day ago"))

pd <- period_bounds(d, "day")
stopifnot(identical(pd$start, d), identical(pd$end, d), pd$days == 1L)
ph <- period_bounds(c("2024-03-05", "2024-08-20"), "half")
stopifnot(identical(ph$start, as.Date(c("2024-01-01", "2024-07-01"))))
stopifnot(identical(ph$end, as.Date(c("2024-06-30", "2024-12-31"))), ph$days == c(182L, 184L))
stopifnot(period_bounds(d, "year")$days == 366L)

stopifnot(identical(p$month_name, c("March", "December")), p$day_of_year == c(65L, 365L))
stopifnot(identical(p$weekend, c(FALSE, TRUE)))
stopifnot(is.na(date_parts(as.Date(NA))$weekend))
stopifnot(identical(date_parts(c("2024-03-05", "2021-01-03"))$week, c(10L, 53L)))
