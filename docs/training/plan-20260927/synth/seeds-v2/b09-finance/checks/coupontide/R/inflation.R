# rebase_index
idx <- c(80, 100, 120)
stopifnot(all.equal(rebase_index(idx, 2001:2003, 2002), c(80, 100, 120)))
stopifnot(all.equal(rebase_index(idx, 2001:2003, 2001), c(100, 125, 150)))
stopifnot(inherits(try(rebase_index(idx, 1:2, 1), silent = TRUE), "try-error"))

# deflate / reflate
stopifnot(all.equal(deflate(c(80, 120), c(80, 120), 1:2, 1), c(80, 80)))
stopifnot(all.equal(reflate(deflate(c(5, 7), c(90, 110), 1:2, 2), c(90, 110), 1:2, 2), c(5, 7)))

# inflation_rate
stopifnot(all.equal(inflation_rate(c(100, 110, 121)), c(NA, 0.1, 0.1)))
stopifnot(all(is.na(inflation_rate(c(1, 2), lag = 2))))

# real_rate / nominal_rate
stopifnot(all.equal(real_rate(0.05, 0.02, approx = TRUE), 0.03))
stopifnot(all.equal(nominal_rate(real_rate(0.05, 0.02), 0.02), 0.05))

# skipped: convert_rate() needs check_rate() from R/tvm.R
