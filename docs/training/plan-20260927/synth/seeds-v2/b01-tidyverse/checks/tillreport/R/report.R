# skipped: needs tibble, lubridate (newSalesReport)
# skipped: needs dplyr, lubridate (buildPeriodReport)

stopifnot(identical(formatCurrency(c(1234.5, -2, NA)), c("EUR 1,234.50", "-EUR 2.00", "-")))
stopifnot(identical(formatCurrency(10, symbol = "$", digits = 0L), "$10"))

stopifnot(identical(formatPercent(c(0.125, NA)), c("12.5%", "-")))
stopifnot(identical(formatPercent(1, digits = 0L), "100%"))

stopifnot(identical(formatDelta(c(0.05, -0.1, NA)), c("+5.0 pp", "-10.0 pp", "-")))

rep_obj <- structure(
  list(
    summary = data.frame(storeId = c("S1", "S2", "S3"), revenue = c(10, 5, 1),
      share = c(0.625, 0.3125, 0.0625)),
    title = "Sales by store",
    from = as.Date("2026-03-01"),
    to = as.Date("2026-03-10"),
    created = Sys.time()
  ),
  class = "salesReport"
)
lines <- format(rep_obj, n = 2L)
stopifnot(lines[1] == "Sales by store")
stopifnot(lines[2] == "Period: 2026-03-01 to 2026-03-10")
stopifnot(lines[length(lines)] == "... 1 more rows")
stopifnot(any(grepl("EUR 10.00", lines, fixed = TRUE)))

out <- utils::capture.output(res <- print(rep_obj))
stopifnot(identical(res, rep_obj), length(out) >= 4)

s <- summary(rep_obj)
stopifnot(s$rows == 3L, s$days == 10L, s$revenue == 16, is.na(s$units))
stopifnot(all.equal(s$revenuePerDay, 1.6))
