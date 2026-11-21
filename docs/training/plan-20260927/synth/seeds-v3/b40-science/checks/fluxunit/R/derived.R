near <- function(a, b, tol = 1e-6) all(abs(a - b) < tol)

stopifnot(near(saturation_vapour_pressure(0, "magnus"), 6.1094))
stopifnot(near(saturation_vapour_pressure(0, "tetens"), 6.1078))
stopifnot(near(saturation_vapour_pressure(0, "bolton"), 6.112))
stopifnot(near(saturation_vapour_pressure(0, "buck"), 6.1121))
stopifnot(abs(saturation_vapour_pressure(20, "buck") - 23.37) < 0.05)
stopifnot(abs(saturation_vapour_pressure(20) - 23.37) < 0.05)

w <- wind_to_components(c(10, 5), c(270, 0))
stopifnot(near(w$u, c(10, 0)), near(w$v, c(0, -5)))
b <- components_to_wind(w$u, w$v)
stopifnot(near(b$speed, c(10, 5)), near(b$direction, c(270, 0)))
stopifnot(near(components_to_wind(0, -3)$direction, 0))

d <- data.frame(temp = c(20, 100, NA), rh = c(50, 101, 40), wind_speed = c(1, -1, 2),
                wind_dir = c(0, 400, 10), pressure = c(1000, 700, 1200))
r <- met_range_report(d)
stopifnot(identical(r$column, c("temp", "rh", "wind_speed", "wind_dir", "pressure")))
stopifnot(identical(r$n_high, c(1L, 1L, 0L, 1L, 1L)), identical(r$n_low, c(0L, 0L, 1L, 0L, 1L)))
stopifnot(r$n_missing[1] == 1L, nrow(met_range_report(d["rh"])) == 1L)

m <- derive_met(data.frame(temp = 20, rh = 100, wind_speed = 10, wind_dir = 90))
stopifnot(near(m$vapour_pressure, saturation_vapour_pressure(20)))
stopifnot(near(m$dew_point, 20), near(m$u, -10), near(m$v, 0))
stopifnot(abs(m$abs_humidity - 17.3) < 0.1)
stopifnot(identical(names(derive_met(data.frame(temp = 1, rh = 1), "dew_point")),
                    c("temp", "rh", "dew_point")))
stopifnot(inherits(try(derive_met(data.frame(temp = 1), "u"), silent = TRUE), "try-error"))
