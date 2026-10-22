raw <- c(1:20)
ref <- 2 + 3 * raw

# newSensorCal / fitLinearCal
lc <- fitLinearCal(raw, ref)
stopifnot(inherits(lc, "sensor_cal"), abs(lc$coef[["slope"]] - 3) < 1e-9, lc$nObs == 20)
stopifnot(inherits(try(fitLinearCal(1:5, 1:5), silent = TRUE), "try-error"))

# fitTwoPointCal
tp <- fitTwoPointCal(rawZero = c(1, 1), rawSpan = 11, refZero = 0, refSpan = 100)
stopifnot(tp$coef[["slope"]] == 10, tp$coef[["intercept"]] == -10)
stopifnot(inherits(try(fitTwoPointCal(5, 5, 0, 10), silent = TRUE), "try-error"))

# fitHumidityCal recovers a known growth model
rh <- rep(c(40, 60, 80, 90), 5)
growth <- 1 + (0.4 / 1.65) / (100 / rh - 1)
hc <- fitHumidityCal(raw * growth, 2 * raw, rh)
stopifnot(abs(hc$coef[["slope"]] - 2) < 1e-9, hc$type == "humidity")

# applyCal
stopifnot(all(abs(applyCal(lc, c(0, 1)) - c(2, 5)) < 1e-9))
stopifnot(applyCal(tp, 0) == 0)
stopifnot(inherits(try(applyCal(hc, 1), silent = TRUE), "try-error"))

# correctDrift
tt <- as.POSIXct("2026-01-01", tz = "UTC") + c(0, 50, 100)
cd <- correctDrift(tt, c(10, 10, 10), tt[c(1, 3)], c(0, 2))
stopifnot(identical(cd, c(10, 9, 8)))
stopifnot(inherits(try(correctDrift(tt, 1:3, tt[c(1, 1)], c(0, 1)), silent = TRUE), "try-error"))

# calMetrics
m <- calMetrics(c(1, 2, 3, NA), c(1, 2, 4, 5))
stopifnot(m[["n"]] == 3, abs(m[["bias"]] + 1 / 3) < 1e-12, is.na(calMetrics(1, 1)[["r2"]]))

# print.sensor_cal
out <- capture.output(res <- print(lc))
stopifnot(identical(res, lc), grepl("linear", out[1]), length(out) == 3)

# crossValidateCal
set.seed(1)
cv <- crossValidateCal(1:60, 2 * (1:60) + stats::rnorm(60), folds = 3L, types = "linear")
stopifnot(nrow(cv) == 3, all(cv$type == "linear"), all(cv$r2 > 0.9))
