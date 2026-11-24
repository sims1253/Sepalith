# skipped: needs future.apply (run.future)
s <- seed.streams(5, seed = 3)
stopifnot(length(s) == 5L, is.integer(s), identical(s, seed.streams(5, seed = 3)))

model <- function(params, days) data.frame(time = 0:days, x = c(0, cumsum(rnorm(days, params$m))))
runs <- replicate.runs(model, list(m = 1), n.reps = 4L, seed = 3, days = 10L)
stopifnot(length(runs) == 4L, identical(attr(runs, "seeds"), s[1:4]), nrow(runs[[1]]) == 11L)
set.seed(s[[2]])
stopifnot(identical(runs[[2]], model(list(m = 1), 10L)))
stopifnot(inherits(try(replicate.runs(1, list()), silent = TRUE), "try-error"))
msg <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
stopifnot(msg(replicate.runs(model, list(),
  n.reps = 0)) == "`n.reps` must be a single positive number.")
stopifnot(msg(replicate.runs(model, list(),
  seed = 0)) == "`seed` must be a single positive number.")
stopifnot(msg(replicate.runs(model, list(),
  days = 0)) == "`days` must be a single positive number.")
stopifnot(msg(run.parallel(model, list(),
  days = -1)) == "`days` must be a single positive number.")

par <- run.parallel(model, list(m = 1), n.reps = 4L, seed = 3, days = 10L, cores = 1L)
stopifnot(identical(par, runs))
stopifnot(inherits(try(run.parallel("model", list()), silent = TRUE), "try-error"))

set.seed(10)
x <- rnorm(200, mean = 5)
for (st in c("mean", "median", "sd")) {
  for (m in c("percentile", "basic", "normal")) {
    ci <- bootstrap.ci(x, st, m, n.boot = 200L)
    stopifnot(identical(names(ci), c("estimate", "lower", "upper")), ci[["lower"]] < ci[["upper"]])
  }
}
ci <- bootstrap.ci(x, "mean", "normal", n.boot = 300L)
stopifnot(ci[["estimate"]] == mean(x), ci[["lower"]] < 5, ci[["upper"]] > 5)
stopifnot(bootstrap.ci(x, "sd")[["estimate"]] == sd(x))

rs <- running.stats(c(2, 4, 6, 8))
stopifnot(identical(names(rs), c("n", "mean", "sd", "lower", "upper")))
stopifnot(identical(rs$mean, c(2, 3, 4, 5)), is.na(rs$sd[1]), isTRUE(all.equal(rs$sd[2], sqrt(2))))
stopifnot(all(rs$lower[-1] < rs$mean[-1]), isTRUE(all.equal(rs$upper - rs$mean,
  rs$mean - rs$lower)))
stopifnot(identical(names(running.stats(1:3, "upper")), c("n", "upper")))

stopifnot(isTRUE(all.equal(mcse(x), sd(x) / sqrt(200))))
b <- mcse(1:20, "batch", n.batches = 4L)
stopifnot(isTRUE(all.equal(b, sd(c(3, 8, 13, 18)) / 2)))
stopifnot(inherits(try(mcse(1:3, "batch", 5L), silent = TRUE), "try-error"))
