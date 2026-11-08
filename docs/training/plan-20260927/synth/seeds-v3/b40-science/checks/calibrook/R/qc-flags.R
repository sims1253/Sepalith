f = c(0L, 1L, NA)
stopifnot(identical(flagLabels(f), c("pass", "fail", "missing")))
stopifnot(identical(flagLabels(f, "traffic"), c("green", "red", "grey")))
stopifnot(identical(flagLabels(f, "letter"), c("P", "F", "M")))
stopifnot(identical(flagLabels(f, "symbol"), c(".", "x", "?")))

a = c(0L, 1L, 1L, NA, NA)
b = c(0L, 0L, 1L, 1L, NA)
cc = c(1L, 0L, 1L, NA, NA)
stopifnot(identical(combineFlags(a, b, cc), c(1L, 1L, 1L, 1L, NA)))
stopifnot(identical(combineFlags(a, b, cc, rule = "majority"), c(0L, 0L, 1L, 1L, NA)))
stopifnot(identical(combineFlags(a, b, cc, rule = "all"), c(0L, 0L, 1L, 1L, NA)))
stopifnot(identical(combineFlags(a, b, cc, rule = "first"), c(0L, 1L, 1L, 1L, NA)))

d = data.frame(dew_point = c(5, 12), temp = c(10, 10), temp_min = c(1, 5), temp_max = c(2, 4),
               gust = c(3, 1), wind_speed = c(2, 0), precip = c(1, 0), rh = c(40, 90),
               wind_dir = c(90, 180))
x = crossCheck(d)
stopifnot(identical(names(x), c("dew_above_temp", "min_above_max", "gust_below_speed",
                                "rain_when_dry", "calm_with_direction")))
stopifnot(identical(x$dew_above_temp, c(0L, 1L)), identical(x$min_above_max, c(0L, 1L)))
stopifnot(identical(x$gust_below_speed, c(0L, 0L)), identical(x$rain_when_dry, c(1L, 0L)))
stopifnot(identical(x$calm_with_direction, c(0L, 1L)))
stopifnot(identical(names(crossCheck(d[c("temp", "dew_point")])), "dew_above_temp"))
stopifnot(crossCheck(d, tol = 3)$dew_above_temp[2] == 0L)

s = flagSummary(list(temp = c(0L, 1L, NA, 0L), rh = c(NA, NA)))
stopifnot(identical(s$check, c("temp", "rh")), identical(s$nPass, c(2L, 0L)))
stopifnot(identical(s$nFail, c(1L, 0L)), identical(s$nMissing, c(1L, 2L)))
stopifnot(abs(s$pctFail[1] - 100 / 3) < 1e-9, is.na(s$pctFail[2]))

m = maskFlagged(data.frame(temp = c(1, 2, 3), rh = c(4, 5, 6)),
                data.frame(temp = c(0L, 1L, NA), other = 1L))
stopifnot(identical(m$temp, c(1, NA, 3)), identical(m$rh, c(4, 5, 6)))
stopifnot(identical(attr(m, "masked"), c(temp = 1L)))
stopifnot(inherits(try(maskFlagged(data.frame(a = 1:2), data.frame(a = 1L)), silent = TRUE),
                   "try-error"))

e = flagEpisodes(c(0L, 1L, 1L, NA, 1L, 0L))
stopifnot(identical(e$start, c(2L, 5L)), identical(e$end, c(3L, 5L)),
          identical(e$length, c(2L, 1L)))
et = flagEpisodes(c(1L, 0L), time = c(10, 20))
stopifnot(et$start == 10, et$end == 10, nrow(flagEpisodes(c(0L, NA))) == 0L)
s2 = flagSummary(data.frame(a = c(1L, 0L, 1L, 1L), b = c(0L, 0L, NA, 0L)))
stopifnot(identical(s2$firstFail, c(1L, NA)), identical(s2$lastFail, c(4L, NA)))
stopifnot(inherits(try(flagSummary(list(c(0L, 1L))), silent = TRUE), "try-error"))
