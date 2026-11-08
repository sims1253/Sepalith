tm <- as.POSIXct("2024-01-01", tz = "UTC") + c(0, 3600, 7200, 0, 7200, 10800)
d <- data.frame(site = c("b", "b", "b", "a", "a", "a"), time = tm,
                pm = c(10, 30, 20, 5, NA, 60))

s <- site.summary(d, "pm")
stopifnot(identical(names(s), c("site", "n", "mean", "sd", "min", "median", "max")))
stopifnot(identical(s$site, c("a", "b")), identical(s$n, c(2L, 3L)))
stopifnot(s$mean == c(32.5, 20), s$sd[2] == 10, s$min == c(5, 10))
stopifnot(s$median == c(32.5, 20), s$max == c(60, 30))
stopifnot(identical(names(site.summary(d, "pm", stats = c("max", "n"))), c("site", "n", "max")))

cpl <- site.completeness(d, "pm")
stopifnot(identical(cpl$site, c("a", "b")), identical(cpl$expected, c(4L, 3L)))
stopifnot(identical(cpl$received, c(3L, 3L)), identical(cpl$valid, c(2L, 3L)))
stopifnot(cpl$completeness == c(50, 100), cpl$last[1] == tm[6], cpl$max_gap_h == c(2, 1))

ex <- site.extremes(d, "pm")
stopifnot(ex$min == c(5, 10), ex$max == c(60, 30))
stopifnot(ex$min_time[1] == tm[4], ex$max_time[1] == tm[6], ex$max_time[2] == tm[2])
ex2 <- site.extremes(rbind(d, data.frame(site = "c", time = tm[1], pm = NA)), "pm")
stopifnot(nrow(ex2) == 3L, is.na(ex2$min[3]))

e <- site.exceedances(d, "pm", c(alert = 15, alarm = 50))
stopifnot(identical(names(e), c("site", "n_valid", "n_alert", "pct_alert", "n_alarm",
                                "pct_alarm")))
stopifnot(identical(e$n_valid, c(2L, 3L)), e$pct_alert == c(50, 200 / 3))
stopifnot(identical(e$n_alert, c(1L, 2L)), identical(e$n_alarm, c(1L, 0L)))
stopifnot(identical(site.exceedances(d, "pm", c(low = 15), direction = "below")$n_low,
                    c(1L, 1L)))
stopifnot(inherits(try(site.exceedances(d, "pm", 15), silent = TRUE), "try-error"))

sd2 <- data.frame(site = c("a", "a", "b"), pm = c(1, 3, 5),
                  time = as.POSIXct(c("2024-01-10", "2024-07-10", "2024-01-10"), tz = "UTC"))
se <- site.seasonal(sd2, "pm")
stopifnot(identical(names(se), c("site", "DJF", "MAM", "JJA", "SON")))
stopifnot(se$DJF == c(1, 5), se$JJA[1] == 3, is.na(se$JJA[2]))
stopifnot(identical(names(site.seasonal(sd2, "pm", hemisphere = "south"))[2], "summer"))
stopifnot(inherits(try(site.exceedances(d, "zz", c(a = 1)), silent = TRUE), "try-error"))
