near = function(a, b, tol = 1e-8) all(abs(a - b) < tol)

stopifnot(is.null(checkDriftInputs(c(1, 2), c(0, 1))))
stopifnot(inherits(try(checkDriftInputs("a", 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(checkDriftInputs(1, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(checkDriftInputs(c(1, 2), c("a", "b")), silent = TRUE), "try-error"))
stopifnot(inherits(try(checkDriftInputs(c(1, 2), 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(checkDriftInputs(c(1, 2), c(2, 1)), silent = TRUE), "try-error"))

x = c(10, 10, 10, 10, 10)
tt = 0:4
stopifnot(near(driftCorrect(x, tt, 0, 2), c(10, 9.5, 9, 8.5, 8)))
stopifnot(near(driftCorrect(x, tt, 0, 2, "step"), c(10, 10, 8, 8, 8)))
ex = driftCorrect(x, tt, 0, 2, "exponential")
stopifnot(near(ex[c(1, 5)], c(10, 8)), ex[2] < 9.5)
stopifnot(near(driftCorrect(x, tt, 0, 2, "end"), rep(8, 5)))

tm = as.POSIXct("2024-01-01", tz = "UTC") + (0:3) * 86400
stopifnot(near(driftRate(c(0, 1, 2, 3), tm), 1))
stopifnot(near(driftRate(c(0, 1, 2, 3), tm, "hour"), 1 / 24))
stopifnot(near(driftRate(c(0, 1, 2, 3), tm, "week"), 7))
stopifnot(near(driftRate(c(0, 1, 2, 3), tm, "month"), 30))
stopifnot(near(driftRate(c(0, 1, 2, 3), tm, "year"), 365))

z = zeroSpanCheck(zero = 2, span = 402, spanTarget = 400)
stopifnot(z$zeroError == 2, !z$zeroPass, near(z$spanError, 0.005), z$spanPass)
stopifnot(z$gain == 1, z$offset == -2)
stopifnot(zeroSpanCheck(0.5, 300, 400, spanTol = 0.1)$spanPass == FALSE)
stopifnot(inherits(try(zeroSpanCheck(5, 1, 1), silent = TRUE), "try-error"))

d = data.frame(time = tm, co2 = c(400, 401, 402, 403), temp = c(NA, NA, NA, 1))
s = driftSummary(d, c("co2", "temp"))
stopifnot(identical(s$sensor, c("co2", "temp")), s$first[1] == 400, s$last[1] == 403)
stopifnot(s$change[1] == 3, near(s$rate[1], 1), is.na(s$rate[2]))
stopifnot(inherits(try(driftSummary(d, "rh"), silent = TRUE), "try-error"))
stopifnot(s$span[1] == 3, is.na(s$span[2]))
stopifnot(inherits(try(checkDriftInputs(c(1, 2), c(1, NA)), silent = TRUE), "try-error"))
