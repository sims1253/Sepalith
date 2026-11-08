stopifnot(identical(parse_dates(c("2024-03-05", "05/03/2024", NA)),
  as.Date(c("2024-03-05", NA, NA))))
stopifnot(parse_dates(" 2024-12-31 ") == as.Date("2024-12-31"))
stopifnot(identical(parse_dates(c("5.3.2024", "05/03/2024", "2024-03-05"), "dmy"),
  as.Date(c("2024-03-05", "2024-03-05", NA))))
stopifnot(identical(parse_dates(c("3/5/2024", "3.5.2024"), "mdy"), as.Date(c("2024-03-05", NA))))
stopifnot(identical(parse_dates(c("20240305", "2024035"), "compact"),
  as.Date(c("2024-03-05", NA))))
stopifnot(identical(parse_dates(c("2024-065", "2024-65"), "ordinal"),
  as.Date(c("2024-03-05", NA))))

stopifnot(parse_dates("2024-03-05") == as.Date("2024-03-05"))
stopifnot(parse_dates("05.03.2024", "dmy") == as.Date("2024-03-05"))
stopifnot(parse_dates("03/05/2024", "mdy") == as.Date("2024-03-05"))
stopifnot(parse_dates("20240305", "compact") == as.Date("2024-03-05"))
stopifnot(parse_dates("2024-065", "ordinal") == as.Date("2024-03-05"))
stopifnot(is.na(parse_dates("garbage", "iso")))
stopifnot(inherits(try(parse_dates(c("2024-01-01", "x"), strict = TRUE), silent = TRUE),
  "try-error"))
stopifnot(!is.na(parse_dates(c("2024-01-01", ""), strict = TRUE)[1]))

rp <- date_format_report(c("2024-02-30", "2024-03-05", "20240101"))
stopifnot(identical(names(rp), c("format", "matched", "parsed", "invalid")))
stopifnot(identical(rp$matched, c(2L, 0L, 0L, 1L, 0L)), rp$invalid[1] == 1L)
stopifnot(nrow(date_format_report("x", c("iso", "dmy"))) == 2L)

stopifnot(guess_date_format(c("20240101", "20240102", "2024-01-03")) == "compact")
stopifnot(guess_date_format(c("01/02/2024", "13/02/2024")) == "dmy")
stopifnot(guess_date_format(c("01/02/2024", "02/02/2024")) == "dmy")
stopifnot(is.na(guess_date_format(c("hello", NA))))

sp <- split_date_parts(c("2024-02-30", "2024-03-05", "x"))
stopifnot(identical(names(sp), c("input", "year", "month", "day", "valid")))
stopifnot(identical(sp$day, c(30L, 5L, NA)), identical(sp$valid, c(FALSE, TRUE, FALSE)))
stopifnot(split_date_parts("7.3.2024", "dmy")$month == 3L)
stopifnot(split_date_parts("3/7/2024", "mdy")$day == 7L)
stopifnot(split_date_parts("20241231", "compact")$valid)
op <- split_date_parts("2024-366", "ordinal")
stopifnot(op$month == 12L, op$day == 31L)

m <- parse_mixed_dates(c("2024-03-05", "20240306", "07.03.2024", "2024-068", "nope", NA))
stopifnot(identical(m$dates[1:4], as.Date(c("2024-03-05", "2024-03-06", "2024-03-07",
  "2024-03-08"))))
stopifnot(identical(m$format, c("iso", "compact", "dmy", "ordinal", NA, NA)))
stopifnot(m$n_failed == 1L)

df <- data.frame(id = 1:3, start = c("2024-01-05", "bad", ""), end = c("2024-02-01", NA,
  "2024-03-01"))
pd <- parse_date_columns(df, c("start", "end"))
stopifnot(inherits(pd$start, "Date"), is.na(pd$start[2]), pd$end[3] == as.Date("2024-03-01"))
stopifnot(identical(attr(pd, "parse_failures"), c(start = 1L, end = 0L)))
pd2 <- parse_date_columns(df, "start", suffix = "_date")
stopifnot(is.character(pd2$start), inherits(pd2$start_date, "Date"))
stopifnot(inherits(try(parse_date_columns(df, "nope"), silent = TRUE), "try-error"))
