near <- function(a, b, tol = 1e-8) all(abs(a - b) < tol)

stopifnot(unit_factors("length")[["km"]] == 1000)
stopifnot(near(unit_factors("mass")[["lb"]], 0.45359237))
stopifnot(unit_factors("volume")[["m3"]] == 1000)
stopifnot(near(unit_factors("pressure")[["mmHg"]] * 760, 101325))
stopifnot(unit_factors("energy")[["kWh"]] == 3.6e6)
stopifnot(unit_factors("time")[["d"]] == 86400)

stopifnot(near(convert_units(1, "mile", "km"), 1.609344))
stopifnot(near(convert_units(c(1, 2), "kg", "g", "mass"), c(1000, 2000)))
stopifnot(inherits(try(convert_units(1, "parsec", "m"), silent = TRUE), "try-error"))
stopifnot(inherits(try(convert_units(1, "m", c("m", "km")), silent = TRUE), "try-error"))
stopifnot(inherits(try(convert_units("1", "m", "km"), silent = TRUE), "try-error"))

stopifnot(near(convert_temperature(100, "C", "F"), 212))
stopifnot(near(convert_temperature(32, "F", "K"), 273.15))
stopifnot(near(convert_temperature(0, "K", "R"), 0))
stopifnot(near(convert_temperature(491.67, "R", "C"), 0))
stopifnot(convert_temperature(5, "C", "C") == 5)

stopifnot(near(convert_rate(1, "m3/s", "l/s", "flow"), 1000))
stopifnot(near(convert_rate(36, "km/h", "m/s", "speed"), 10))
stopifnot(near(convert_rate(1, "kg/h", "g/min", "discharge"), 1000 / 60))
stopifnot(near(convert_rate(1, "kWh/h", "J/s", "power"), 1000))
stopifnot(inherits(try(convert_rate(1, "m3", "l/s"), silent = TRUE), "try-error"))

d <- data.frame(depth = c(1, 2), temp = c(0, 100), id = 1:2)
out <- convert_columns(d, c(depth = "m", temp = "C"), c(depth = "cm", temp = "F"),
                       c(depth = "length", temp = "temperature"))
stopifnot(near(out$depth, c(100, 200)), near(out$temp, c(32, 212)))
stopifnot(identical(attr(out, "units"), c(depth = "cm", temp = "F")))
stopifnot(inherits(try(convert_columns(d, c(z = "m"), c(z = "cm"), c(z = "length")),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(convert_columns(d, c(depth = "m"), c(x = "cm"), c(depth = "length")),
                       silent = TRUE), "try-error"))

p <- pretty_unit(c(2500, 3000, 4000))
stopifnot(p$unit == "mile", near(p$value, c(2500, 3000, 4000) / 1609.344))
stopifnot(pretty_unit(c(2500, 3000), candidates = c("m", "km"))$unit == "km")
stopifnot(pretty_unit(1e-5, candidates = c("mm", "m"))$unit == "mm")
