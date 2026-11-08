bp <- aqi_breakpoints("pm25")
stopifnot(nrow(bp) == 6L, identical(names(bp), c("conc_lo", "conc_hi", "index_lo", "index_hi")))
stopifnot(bp$index_lo[2] == 51, bp$conc_hi[6] == 500.4)
stopifnot(aqi_breakpoints("pm10")$conc_hi[1] == 54, aqi_breakpoints("o3")$conc_hi[2] == 70)
stopifnot(aqi_breakpoints("no2")$conc_lo[3] == 100, aqi_breakpoints("so2")$conc_hi[1] == 35)
stopifnot(aqi_breakpoints("co")$conc_hi[6] == 50.4)

stopifnot(identical(aqi_subindex(c(6, 12, NA, 900), "pm25"), c(25, 51, NA, 500)))
stopifnot(aqi_subindex(-1, "pm10") |> is.na())

stopifnot(identical(check_concentration(c(1, 2), "pm25", 2L), c(1, 2)))
stopifnot(inherits(try(check_concentration("a", "o3", 1L), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_concentration(1, "o3", 2L), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_concentration(-1, "o3", 1L), silent = TRUE), "try-error"))

a <- air_quality_index(pm25 = c(6, 40), pm10 = c(27, 10), o3 = c(10, 10), no2 = c(1, 1),
                       so2 = c(1, 1))
stopifnot(identical(a$sub_pm25, c(25, 112)), a$sub_pm10[1] == 25)
stopifnot(identical(a$aqi, c(25, 112)), a$dominant[2] == "pm25")
stopifnot(!is.na(a$sub_o3[1]), !is.na(a$sub_no2[1]), !is.na(a$sub_so2[1]))
b <- air_quality_index(o3 = 60)
stopifnot(is.na(b$sub_pm25), b$dominant == "o3", b$aqi == b$sub_o3)
stopifnot(inherits(try(air_quality_index(), silent = TRUE), "try-error"))
stopifnot(inherits(try(air_quality_index(pm25 = 1, so2 = c(1, 2)), silent = TRUE), "try-error"))

ci <- comfort_index(temp = c(20, 29), humidity = c(50, 75), wind = c(1, 10))
stopifnot(ci$comfort[1] == 1, ci$score_temp[2] == 0.5, ci$score_humidity[2] == 0.5)
stopifnot(ci$score_wind[2] == 0, abs(ci$comfort[2] - (0.25 + 0.15)) < 1e-12)
