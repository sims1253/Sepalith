stopifnot(identical(parseCountInput(" 12 ", "N"), list(value = 12L, error = NULL)))
stopifnot(identical(parseCountInput("", "N")$error, "N is required."))
stopifnot(identical(parseCountInput("1.5", "N")$error, "N must be a whole number."))
stopifnot(identical(parseCountInput("20", "N", max = 10)$error, "N must be between 0 and 10."))

stopifnot(identical(parseRateInput("0.4", "Rate")$value, 0.4))
stopifnot(identical(parseRateInput("35%", "Rate")$value, 0.35))
stopifnot(identical(parseRateInput("abc", "Rate")$error, "Rate must be a number."))
stopifnot(identical(parseRateInput("150%", "Rate")$error, "Rate must be between 0 and 1."))
stopifnot(identical(parseRateInput(NA, "Rate")$error, "Rate is required."))

stopifnot(identical(parseDateInput("2024-05-01", "Day")$value, as.Date("2024-05-01")))
stopifnot(identical(parseDateInput("01/05/2024", "Day")$error,
                    "Day must be a date (YYYY-MM-DD)."))
stopifnot(identical(parseDateInput("1999-01-01", "Day")$error,
                    "Day must be between 2000-01-01 and 2099-12-31."))
stopifnot(identical(parseDateInput("  ", "Day")$error, "Day is required."))

stopifnot(identical(parseIdInput("ab123", "ID")$value, "AB123"))
stopifnot(identical(parseIdInput("A1", "ID")$error, "ID must match ^[A-Z]{2}[0-9]{3}$."))
stopifnot(identical(parseIdInput(character(0), "ID")$error, "ID is required."))

ok = checkWaveSettings("wv001", "2024-03-01", "2024-03-31", "500", "40%")
stopifnot(length(ok$errors) == 0L)
stopifnot(identical(names(ok$values), c("waveId", "startDate", "endDate", "targetN",
                                        "responseRate")))
stopifnot(identical(ok$values$targetN, 500L), identical(ok$values$responseRate, 0.4))
bad = checkWaveSettings("", "2024-03-31", "2024-03-01", "0", "x")
stopifnot(identical(names(bad$errors), c("waveId", "targetN", "responseRate", "endDate")))
stopifnot(identical(bad$errors[["endDate"]], "End date must not be before Start date."))
stopifnot(identical(bad$errors[["targetN"]], "Target interviews must be between 1 and Inf."))
bad2 = checkWaveSettings("WV001", "x", "y", "10", "0.5")
stopifnot(identical(names(bad2$errors), c("startDate", "endDate")))

q = data.frame(cell = c("a", "b"), target = c(10, 20), achieved = c(5, 25), weight = c(1, 1))
stopifnot(length(checkQuotaTable(q)) == 0L)
stopifnot(identical(checkQuotaTable(q[, 1:2]),
                    c("Missing column: achieved.", "Missing column: weight.")))
q2 = q
q2$cell = c("a", "a")
q2$target = c("x", "y")
q2$achieved = c(-1, 2)
q2$weight = c(1, -2)
stopifnot(identical(checkQuotaTable(q2), c(
  "Column cell must be unique and complete.",
  "Column target must be non-negative numbers.",
  "Column achieved must be non-negative numbers.",
  "Column weight must be non-negative numbers."
)))

cs = checkContactSettings("5", "24", "9", "18")
stopifnot(length(cs$errors) == 0L, identical(cs$values$windowEnd, 18L))
stopifnot(identical(names(cs$values), c("maxAttempts", "minGapHours", "windowStart", "windowEnd")))
cb = checkContactSettings("0", "200", "x", "")
stopifnot(identical(names(cb$errors), c("maxAttempts", "minGapHours", "windowStart", "windowEnd")))
stopifnot(identical(cb$errors[["maxAttempts"]], "Max attempts must be between 1 and 20."))
stopifnot(identical(cb$errors[["minGapHours"]], "Minimum gap must be between 0 and 168."))
stopifnot(identical(cb$errors[["windowEnd"]], "Call window end is required."))
stopifnot(identical(checkContactSettings("3", "2", "18", "9")$errors[["windowEnd"]],
                    "Call window end must be after Call window start."))
