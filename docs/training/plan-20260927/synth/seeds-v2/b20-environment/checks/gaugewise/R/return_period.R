# water_year
d <- as.Date(c("2024-09-30", "2024-10-01", "2025-03-15"))
stopifnot(identical(water_year(d), c(2024L, 2025L, 2025L)))
stopifnot(identical(water_year(d, 1L), c(2024L, 2024L, 2025L)))
stopifnot(inherits(try(water_year("2024-01-01"), silent = TRUE), "try-error"))

# annual_maxima skips patchy and all-missing years
dts <- seq(as.Date("2020-10-01"), as.Date("2022-09-30"), by = "day")
fl <- rep(1, length(dts))
fl[100] <- 50
fl[dts >= as.Date("2021-10-01")] <- NA
am <- annual_maxima(dts, fl, max_missing = 1)
stopifnot(nrow(am) == 1, am$peak_flow == 50, am$peak_date == dts[100])

# fit_gumbel / fit_lp3 / fit_gev
set.seed(3)
peaks <- 100 - 20 * log(-log(stats::runif(200)))
gu <- fit_gumbel(peaks)
stopifnot(inherits(gu, "gw_fit"), abs(gu$params[["scale"]] - 20) < 4)
lp <- fit_lp3(peaks)
stopifnot(lp$dist == "lp3", lp$n == 200)
stopifnot(inherits(try(fit_lp3(c(-1, rep(1, 10))), silent = TRUE), "try-error"))
ge <- fit_gev(peaks)
stopifnot(abs(ge$params[["shape"]]) < 0.2, abs(ge$params[["location"]] - 100) < 10)

# design_flow
df <- design_flow(gu, c(2, 100))
stopifnot(df$discharge[2] > df$discharge[1], df$probability[2] == 0.99)
g0 <- gu
g0$dist <- "gev"
g0$params <- c(location = 100, scale = 20, shape = 0)
stopifnot(isTRUE(all.equal(design_flow(g0, 10)$discharge, design_flow(gu, 10)$discharge,
  tolerance = 0.2)))
stopifnot(is.finite(design_flow(g0, 10)$discharge))
stopifnot(inherits(try(design_flow(gu, 1), silent = TRUE), "try-error"))

# print.gw_fit
out <- capture.output(res <- print(gu))
stopifnot(identical(res, gu), grepl("gumbel", out[1]), length(out) == 3)

# design_flow_ci
ci <- design_flow_ci(peaks, "gumbel", return_period = 10, n_boot = 50, seed = 1)
stopifnot(ci$lower < ci$estimate, ci$estimate < ci$upper, ci$n_failed == 0L)
