# unit_factors / convert_linear
stopifnot(unit_factors("precipitation")[["inch"]] == 25.4)
stopifnot(inherits(try(unit_factors("mass"), silent = TRUE), "try-error"))
bad <- try(convert_linear(1, "mm", "ft", "precipitation"), silent = TRUE)
stopifnot(inherits(bad, "try-error"))

# factor-based converters
stopifnot(abs(convert_speed(10, "m_s", "km_h") - 36) < 1e-12)
stopifnot(convert_pressure(1013, "hPa", "kPa") == 101.3)
stopifnot(convert_precipitation(c(1, NA))[1] == 25.4)
stopifnot(abs(convert_discharge(1000, "L_s", "m3_s") - 1) < 1e-12)

# convert_temperature
stopifnot(convert_temperature(212, "F", "C") == 100, convert_temperature(0, "C", "K") == 273.15)
lowk <- suppressWarnings(convert_temperature(-300, "C", "K"))
stopifnot(is.na(lowk))

# convert_columns
df <- data.frame(wind = 10, rain = 1)
spec <- list(wind = c("speed", "m_s", "km_h"), rain = c("precipitation", "inch", "mm"))
cc <- convert_columns(df, spec)
stopifnot(abs(cc$wind - 36) < 1e-12, cc$rain == 25.4, attr(cc, "units")$rain == "mm")

# harmonise_units, including a row with a missing station id
obs <- data.frame(station = c("A", "B", NA), temp = c(32, 0, 5))
meta <- data.frame(
  station = c("A", "B"), variable = "temp", quantity = "temperature", unit = c("F", "C")
)
hu <- suppressWarnings(harmonise_units(obs, meta, c(temp = "C")))
stopifnot(identical(hu$temp, c(0, 0, 5)), identical(attr(hu, "units"), list(temp = "C")))
