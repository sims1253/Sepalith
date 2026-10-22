flow <- c(5, 1, 3, 2, 4, NA)

# fdc_exceedance
ex <- fdc_exceedance(flow)
stopifnot(identical(ex$flow, c(5, 4, 3, 2, 1)), identical(ex$exceedance, (1:5) / 6))
stopifnot(inherits(try(fdc_exceedance(flow, na_rm = FALSE), silent = TRUE), "try-error"))
stopifnot(abs(fdc_exceedance(1:4, "cunnane")$exceedance[1] - 0.6 / 4.2) < 1e-12)

# fdc_quantile and the Q5 / Q50 / Q95 wrappers
q <- fdc_quantile(flow, c(50, 1))
stopifnot(q[["Q50"]] == 3, q[["Q1"]] == 5)
stopifnot(fdc_q50(flow) == 3, fdc_q5(flow) == 5, fdc_q95(flow) == 1)
stopifnot(inherits(try(fdc_quantile(flow, 100), silent = TRUE), "try-error"))

# fdc_slope is negative for a decreasing curve
stopifnot(fdc_slope(1:100) < 0)
stopifnot(inherits(try(fdc_slope(1:10, 60, 30), silent = TRUE), "try-error"))

# fdc_summary
d <- data.frame(station = rep(c("a", "b"), c(20, 5)), flow = c(1:20, 1:5))
fsum <- fdc_summary(d, min_days = 10)
stopifnot(identical(fsum$group, c("a", "b")), fsum$n_days[2] == 5, is.na(fsum$q50[2]))
stopifnot(fsum$q50[1] == 10.5)

# fdc_compare
cmp <- fdc_compare(1:100, 2 * (1:100))
stopifnot(all(abs(cmp$curve$relative - 1) < 1e-12), all(abs(cmp$segments - 1) < 1e-12))
