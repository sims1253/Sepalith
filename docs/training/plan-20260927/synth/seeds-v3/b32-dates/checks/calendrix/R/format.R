stopifnot(identical(ordinal_day(c(1, 2, 3, 4, 11, 12, 13, 21, 22, 23, 31)),
  c("1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "23rd", "31st")))

x <- as.Date(c("2024-03-05", NA))
stopifnot(identical(format_date_style(x, "iso"), c("2024-03-05", NA)))
stopifnot(format_date_style(x[1], "long") == "5 March 2024")
stopifnot(format_date_style(x[1], "short") == "5 Mar 2024")
stopifnot(identical(format_date_style(x, "compact"), c("20240305", NA)))
stopifnot(format_date_style(x[1], "us") == "March 5, 2024")
stopifnot(format_date_style(x[1], "ordinal") == "5th March 2024")
stopifnot(format_date_style(as.Date("2024-12-30"), "week") == "2025-W01")
stopifnot(format_date_style(as.Date("2024-08-01"), "quarter") == "2024-Q3")

stopifnot(format_duration(as.difftime(93784, units = "secs")) == "1d 2h")
stopifnot(format_duration(as.difftime(93784, units = "secs"), parts = 4L) == "1d 2h 3m 4s")
stopifnot(format_duration(as.difftime(-90, units = "mins")) == "-1h 30m")
stopifnot(identical(format_duration(as.difftime(c(0, NA), units = "secs")), c("0s", NA)))

stopifnot(format_date_range("2024-03-03", "2024-03-09") == "3-9 March 2024")
stopifnot(format_date_range("2024-02-28", "2024-03-03") == "28 February - 3 March 2024")
stopifnot(format_date_range("2024-12-30", "2025-01-02") ==
  "30 December 2024 - 2 January 2025")
stopifnot(format_date_range("2024-03-03", "2024-03-03") == "3 March 2024")
stopifnot(inherits(try(format_date_range("2024-03-04", "2024-03-03"), silent = TRUE),
  "try-error"))

t <- as.POSIXct("2024-03-05 14:27:09", tz = "UTC")
stopifnot(format_timestamp(t) == "2024-03-05 14:27:09")
stopifnot(format_timestamp(t, "minute") == "2024-03-05 14:27")
stopifnot(format_timestamp(t, "hour") == "2024-03-05 14:00")
stopifnot(format_timestamp(t, "day", tz = "Asia/Tokyo") == "2024-03-05")

ref <- as.Date("2024-03-10")
lab <- relative_date_label(ref + c(0, 1, -1, 3, -5, 21, -15, NA), ref)
stopifnot(identical(lab, c("today", "tomorrow", "yesterday", "in 3 days", "5 days ago",
  "in 3 weeks", "2 weeks ago", NA)))
