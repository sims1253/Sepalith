near <- function(a, b, tol = 1e-8) all(abs(a - b) < tol)
tm <- as.POSIXct("2024-01-01", tz = "UTC") + c(0, 86400, 2 * 86400)
d <- data.frame(site = rep(c("a", "b"), each = 3), time = c(tm, tm),
                no2 = c(1, 2, 6, 2, 4, 6))

stopifnot(near(site.anomaly(d, "no2")$anomaly, c(-2, -1, 3, -2, 0, 2)))
stopifnot(near(site.anomaly(d, "no2", baseline = "site.median")$anomaly, c(-1, 0, 4, -2, 0, 2)))
stopifnot(near(site.anomaly(d, "no2", baseline = "overall.mean")$anomaly, d$no2 - 3.5))
ra <- site.anomaly(d, "no2", baseline = "reference", reference = c(a = 1, b = 2))
stopifnot(near(ra$anomaly, c(0, 1, 5, 0, 2, 4)), attr(ra, "baseline") == "reference")
stopifnot(inherits(try(site.anomaly(d, "no2", baseline = "reference", reference = c(a = 1)),
                       silent = TRUE), "try-error"))

ps <- site.pair.stats(d, "no2", "a", "b")
stopifnot(ps$n == 3L, ps$mean_a == 3, ps$mean_b == 4, ps$bias == -1)
stopifnot(near(ps$rmse, sqrt(5 / 3)), ps$r > 0.9)
stopifnot(inherits(try(site.pair.stats(d, "no2", "z", "b"), silent = TRUE), "try-error"))
stopifnot(inherits(try(site.pair.stats(d, "no2", "a", "z"), silent = TRUE), "try-error"))

tr <- site.trend(d, "no2", per = "day")
stopifnot(identical(tr$site, c("a", "b")), near(tr$slope, c(2.5, 2)), tr$n == c(3L, 3L))
stopifnot(near(site.trend(d, "no2", per = "month")$slope[2], 60))
stopifnot(near(site.trend(d, "no2")$slope[2], 2 * 365.25, 1e-6))
stopifnot(near(site.trend(d, "no2", per = "decade")$slope[2], 2 * 3652.5, 1e-6))
stopifnot(is.na(site.trend(d, "no2", min.n = 4L)$slope[1]))
stopifnot(near(tr$r.squared[2], 1), identical(names(tr), c("site", "n", "intercept", "slope",
                                                         "p.value", "r.squared")))

nw <- site.network(d, "no2")
stopifnot(identical(nw$n_sites, c(2L, 2L, 2L)), near(nw$value, c(1.5, 3, 6)))
stopifnot(near(site.network(d, "no2", method = "median")$value, c(1.5, 3, 6)))
stopifnot(near(site.network(d, "no2", method = "trimmed")$value, c(1.5, 3, 6)))
stopifnot(near(site.network(d, "no2", method = "weighted", weights = c(a = 3, b = 1))$value,
               c(1.25, 2.5, 6)))
stopifnot(inherits(try(site.network(d, "no2", method = "weighted"), silent = TRUE),
                   "try-error"))

rk <- site.rank(d, "no2")
stopifnot(identical(rk$site, c("b", "a")), identical(names(rk), c("rank", "site", "mean")))
stopifnot(identical(site.rank(d, "no2", by = "median")$site, c("b", "a")))
stopifnot(site.rank(d, "no2", by = "max")$max[1] == 6)
stopifnot(identical(site.rank(d, "no2", by = "p95", decreasing = FALSE)$rank, 1:2))
