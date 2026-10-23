days_in_month <- function(x) {
  first <- as.Date(format(as.Date(x), "%Y-%m-01"))
  nxt <- as.Date(sapply(first, function(f) seq(f, by = "month", length.out = 2)[2]),
    origin = "1970-01-01")
  as.integer(nxt - first)
}

x <- as.Date("2024-05-15")
stopifnot(start_of_period(x, "week") == as.Date("2024-05-13"))
stopifnot(start_of_period(x, "week", week_start = 7L) == as.Date("2024-05-12"))
stopifnot(start_of_period(x, "month") == as.Date("2024-05-01"))
stopifnot(start_of_period(x, "quarter") == as.Date("2024-04-01"))
stopifnot(start_of_period(x, "year") == as.Date("2024-01-01"))
stopifnot(start_of_period(as.Date("2024-08-20"), "half") == as.Date("2024-07-01"))
stopifnot(inherits(try(start_of_period(x, "week", week_start = 0L), silent = TRUE), "try-error"))

stopifnot(end_of_period(x, "week") == as.Date("2024-05-19"))
stopifnot(end_of_period(x, "week", week_start = 7L) == as.Date("2024-05-18"))
stopifnot(end_of_period(as.Date("2024-02-10"), "month") == as.Date("2024-02-29"))
stopifnot(end_of_period(x, "quarter") == as.Date("2024-06-30"))
stopifnot(end_of_period(x, "year") == as.Date("2024-12-31"))
stopifnot(end_of_period(x, "half") == as.Date("2024-06-30"))
stopifnot(inherits(try(end_of_period(x, "week", week_start = 1:2), silent = TRUE), "try-error"))

stopifnot(identical(is_period_start(as.Date(c("2024-04-01", "2024-04-02")), "quarter"),
  c(TRUE, FALSE)))
stopifnot(is_period_start(as.Date("2024-05-13"), "week"))
stopifnot(is_period_end(as.Date("2024-05-19"), "week"))
stopifnot(!is_period_end(as.Date("2024-12-30"), "year"))

b <- period_bounds(as.Date(c("2024-02-10", "2023-02-28")), "month")
stopifnot(identical(names(b), c("date", "start", "end", "length", "elapsed")))
stopifnot(identical(b$length, c(29L, 28L)), identical(b$elapsed, c(10L, 28L)))

d <- as.Date(c("2024-01-01", "2024-06-01", NA, "2024-12-31"))
cl <- clamp_dates(d, "2024-03-01", "2024-09-30")
stopifnot(identical(cl, as.Date(c("2024-03-01", "2024-06-01", NA, "2024-09-30"))))
stopifnot(identical(clamp_dates(d, upper = "2024-03-01")[1:2],
  as.Date(c("2024-01-01", "2024-03-01"))))
stopifnot(inherits(try(clamp_dates(d, "2025-01-01", "2024-01-01"), silent = TRUE),
  "try-error"))

stopifnot(period_progress(as.Date("2024-05-19"), "week") == 1)
stopifnot(period_progress(as.Date("2024-05-13"), "week", inclusive = FALSE) == 0)
stopifnot(all.equal(period_progress(as.Date("2024-04-15"), "month"), 0.5))
