near = function(a, b, tol = 1e-8) all(abs(a - b) < tol)

dc = defaultCoefficients()
stopifnot(identical(names(dc), c("temp", "rh", "pressure", "co2", "pm25")))
stopifnot(dc$co2[["gain"]] == 1, dc$temp[["offset"]] == 0)

stopifnot(identical(checkCoefficients(dc), dc))
stopifnot(inherits(try(checkCoefficients(list(1)), silent = TRUE), "try-error"))
stopifnot(inherits(try(checkCoefficients(list(temp = c(offset = 0, gain = 1))),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(checkCoefficients(list(temp = c(offset = NA, gain = 1, curve = 0))),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(checkCoefficients(list(temp = c(offset = 0, gain = 0, curve = 0))),
                       silent = TRUE), "try-error"))

raw = data.frame(temp = c(10, 20), rh = c(50, 98), pressure = c(1000, 1010),
                 co2 = c(5, 400), pm25 = c(1, 10), other = 1:2)
cf = list(temp = c(offset = 1, gain = 2, curve = 0),
          rh = c(offset = 5, gain = 1, curve = 0),
          pressure = c(offset = 0, gain = 1, curve = 0.001),
          co2 = c(offset = -10, gain = 1, curve = 0),
          pm25 = c(offset = -2, gain = 1, curve = 0))
out = calibrateSensors(raw, cf)
stopifnot(near(out$temp, c(21, 41)), near(out$rh, c(55, 100)))
stopifnot(near(out$pressure, c(2000, 1010 + 1020.1)), near(out$co2, c(0, 390)))
stopifnot(near(out$pm25, c(0, 8)), identical(out$other, 1:2))
stopifnot(identical(attr(out, "calibrated"), c("temp", "rh", "pressure", "co2", "pm25")))
stopifnot(identical(attr(calibrateSensors(raw, cf, "rh"), "calibrated"), "rh"))
stopifnot(identical(calibrateSensors(raw["temp"])$temp, raw$temp))

obs = c(1, 2, 3, 4, 5)
f1 = fitCalibration(obs + 2, obs, "offset")
stopifnot(near(f1$coefs, c(offset = 2, gain = 1, curve = 0)), near(f1$rmse, 0))
f2 = fitCalibration(obs * 3, obs, "ratio")
stopifnot(near(f2$coefs[["gain"]], 3), f2$model == "ratio")
f3 = fitCalibration(1 + 2 * obs, obs)
stopifnot(near(f3$coefs, c(offset = 1, gain = 2, curve = 0)))
f4 = fitCalibration(1 + 2 * obs + 0.5 * obs^2, obs, "quadratic")
stopifnot(near(f4$coefs[["curve"]], 0.5, 1e-6), f4$rmse < 1e-6)
stopifnot(inherits(try(fitCalibration(1:2, 1:2), silent = TRUE), "try-error"))

d = data.frame(temp = c(1, 2, 3, 4), ref_temp = c(0, 1, 2, 3), rh = 1:4)
rep = calibrationReport(d)
stopifnot(nrow(rep) == 1L, rep$sensor == "temp", rep$bias == 1, rep$rmse == 1, near(rep$r, 1))
stopifnot(inherits(try(checkCoefficients(list(rh = c(offset = 0, gain = 1, curve = Inf))),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(checkCoefficients(list(rh = c(offset = 0, gain = NaN, curve = 0))),
                       silent = TRUE), "try-error"))
