set.seed(10)
tt <- pp.homogeneous(5, 10)
stopifnot(all(diff(tt) > 0), all(tt <= 10), length(tt) > 20)
stopifnot(length(pp.inhomogeneous(function(t) 0 * t, 10, rate.max = 1)) == 0)
stopifnot(inherits(try(pp.inhomogeneous(function(t) 2 + 0 * t, 50, 1), silent = TRUE),
  "try-error"))

cp <- pp.compound(3, 5, mark.fun = function(n) rep(2, n))
stopifnot(identical(names(cp), c("time", "mark", "total")), all(cp$total == 2 * seq_len(nrow(cp))))

w <- pp.window.counts(c(0.1, 0.2, 1.5, 2.9, 3), horizon = 3, width = 1)
stopifnot(identical(w$count, c(2L, 1L, 2L)), identical(w$end, c(1, 2, 3)))
w2 <- pp.window.counts(c(0.1, 0.5), horizon = 0.9, width = 0.3)
stopifnot(nrow(w2) == 3, isTRUE(all.equal(w2$end[3], 0.9)), sum(w2$count) == 2)
w3 <- pp.window.counts(1, horizon = 2.5, width = 1)
stopifnot(nrow(w3) == 3, w3$end[3] == 2.5)

dt <- pp.dispersion.test(c(4, 4, 4, 4), alternative = "under")
stopifnot(dt$statistic == 0, dt$df == 3, dt$p.value < 0.01)
stopifnot(pp.dispersion.test(c(0, 20, 0, 20), alternative = "over")$p.value < 0.001)
stopifnot(inherits(try(pp.dispersion.test(c(0, 0)), silent = TRUE), "try-error"))

re <- pp.rate.estimate(c(0.5, 1.5, 1.6), horizon = 2, width = 1)
stopifnot(identical(re$rate, c(1, 2)), all(re$lower <= re$rate), all(re$upper >= re$rate))
stopifnot(pp.rate.estimate(numeric(0), 1)$lower == 0)
