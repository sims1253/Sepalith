# checkTemps
stopifnot(is.null(checkTemps(1:3, 2:4)))
stopifnot(inherits(try(checkTemps(5, 1), silent = TRUE), "try-error"))

# ddAverage
stopifnot(identical(ddAverage(c(10, 0, 25, NA), c(20, 5, 40, 1)), c(5, 0, 17.5, NA)))

# ddTriangle
stopifnot(ddTriangle(10, 20) == 5, ddTriangle(0, 5) == 0, ddTriangle(31, 35) == 20)
stopifnot(abs(ddTriangle(0, 20) - 2.5) < 1e-12)

# ddSine: symmetric day centred on base gives amp / pi
stopifnot(abs(ddSine(5, 15) - 5 / pi) < 1e-12, ddSine(12, 18) == 5, is.na(ddSine(NA, 1)))

# degreeDays dispatch
stopifnot(identical(degreeDays(10, 20, "triangle"), ddTriangle(10, 20)))
stopifnot(identical(degreeDays(10, 20, base = 12), ddAverage(10, 20, base = 12)))

# accumulateDegreeDays resets at the biofix and handles unsorted input
dts <- as.Date(c("2025-03-02", "2025-02-27", "2025-03-01", "2025-03-03"))
acc <- accumulateDegreeDays(dts, c(2, 5, 1, NA))
stopifnot(identical(acc$season, c(2025L, 2024L, 2025L, 2025L)))
stopifnot(identical(acc$cumDd, c(3, 5, 1, 3)), identical(acc$nMissing, c(0L, 0L, 0L, 1L)))
stopifnot(inherits(try(accumulateDegreeDays(as.Date(NA), 1), silent = TRUE), "try-error"))

# heatingDegreeDays / coolingDegreeDays
stopifnot(identical(heatingDegreeDays(c(10, 20, NA)), c(5.5, 0, NA)))
stopifnot(identical(coolingDegreeDays(c(25, 20)), c(3, 0)))

# chillUnits
stopifnot(identical(chillUnits(c(0, 5, 17, 30, NA)), c(0, 1, -0.5, -1, NA)))
