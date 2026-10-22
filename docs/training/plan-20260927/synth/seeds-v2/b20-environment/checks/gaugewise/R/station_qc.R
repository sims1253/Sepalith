# qc_codes
stopifnot(identical(qc_codes()[["missing"]], 9L))

# qc_range
stopifnot(identical(qc_range(c(-5, 0.5, 5, 50, NA), 0, 40, margin = 1), c(2L, 1L, 0L, 2L, 9L)))

# qc_step
stopifnot(identical(qc_step(c(1, 2, 6, 20, NA), max_step = 3), c(0L, 0L, 1L, 2L, 9L)))

# qc_persistence
pf <- qc_persistence(c(1, 2, 2, 2, 3), min_run = 3L)
stopifnot(identical(pf, c(0L, 1L, 1L, 1L, 0L)), all(qc_persistence(1:5, 2L) == 0L))

# qc_temperature_pair
tp <- qc_temperature_pair(c(1, 5, NA), c(10, 4, 3))
stopifnot(identical(tp$tmin, c(0L, 2L, 9L)), identical(tp$tmax, c(0L, 2L, 0L)))

# qc_combine
stopifnot(identical(qc_combine(c(0L, 1L), c(2L, 0L)), c(2L, 1L)))
stopifnot(inherits(try(qc_combine(1L, 1:2), silent = TRUE), "try-error"))

# qc_station / qc_report
d <- data.frame(tmin = c(1, 2, 30), tmax = c(10, 12, 20))
lim <- list(
  tmin = list(lower = -40, upper = 40, max_step = 50),
  tmax = list(lower = -40, upper = 50, max_step = 50)
)
qs <- qc_station(d, lim)
stopifnot(identical(qs$tmin_qc, c(0L, 0L, 2L)), identical(qs$tmax_qc, c(0L, 0L, 2L)))
rep_ <- qc_report(qs)
stopifnot(identical(rep_$variable, c("tmin", "tmax")), all(rep_$bad == 1L), all(rep_$good == 2L))

# qc_spatial
set.seed(1)
nb <- matrix(rep(1:20, 3) + stats::rnorm(60, sd = 0.1), ncol = 3)
x <- c(1:19, 60)
sp <- qc_spatial(x + stats::rnorm(20, sd = 0.1), nb)
stopifnot(sp[20] == 2L, all(sp[1:19] == 0L))
stopifnot(inherits(try(qc_spatial(1:3, nb), silent = TRUE), "try-error"))
