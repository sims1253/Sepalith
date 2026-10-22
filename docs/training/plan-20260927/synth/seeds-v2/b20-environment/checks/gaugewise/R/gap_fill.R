# find_gaps
g <- find_gaps(c(1, NA, NA, 4, NA))
stopifnot(identical(g$start, c(2L, 5L)), identical(g$length, c(2L, 1L)), nrow(find_gaps(1:3)) == 0)

# fill_linear: interior gaps only
fl <- fill_linear(c(1, NA, NA, 4, NA))
stopifnot(identical(fl$value[1:4], c(1, 2, 3, 4)), is.na(fl$value[5]))
stopifnot(identical(fl$filled, c(FALSE, TRUE, TRUE, FALSE, FALSE)))
stopifnot(!any(fill_linear(c(1, NA, NA, 4), max_gap = 1)$filled))

# fill_climatology
dts <- seq(as.Date("2024-01-01"), by = "day", length.out = 3 * 365)
x <- rep(5, length(dts))
x[c(10, 400)] <- NA
fc <- fill_climatology(x, dts)
stopifnot(all(fc$value == 5), sum(fc$filled) == 2)
stopifnot(inherits(try(fill_climatology(1:3, dts), silent = TRUE), "try-error"))

# fill_neighbour
nbr <- 1:100
y <- 2 * nbr + 1
y[c(5, 50)] <- NA
fn <- suppressWarnings(fill_neighbour(y, nbr))
stopifnot(abs(fn$value[50] - 101) < 1e-9, sum(fn$filled) == 2)
weak <- suppressWarnings(fill_neighbour(y[1:20], nbr[1:20]))
stopifnot(!any(weak$filled))

# fill_cascade / fill_summary
casc <- fill_cascade(c(1, NA, 3, NA, NA, NA, NA, 8), dts[1:8], methods = "linear")
stopifnot(identical(casc$source[1:3], c("observed", "linear", "observed")), is.na(casc$source[4]))
fs <- fill_summary(casc)
stopifnot(fs[["observed"]] == 3L, fs[["linear"]] == 1L, fs[["missing"]] == 4L)

# evaluate_fillers
ev <- evaluate_fillers(sin(seq_along(dts) / 30), dts, n_gaps = 20L, seed = 42)
stopifnot(identical(ev$method, c("linear", "climatology")), ev$n[1] > 0)
stopifnot(ev$rmse[1] < 0.01)
