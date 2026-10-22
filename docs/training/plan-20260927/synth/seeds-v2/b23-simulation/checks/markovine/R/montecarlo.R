# skipped: needs tibble/dplyr: mc.design(), mc.run(), mc.aggregate(),
# mc.reps.needed()
ok <- mc.run.one(function(a) c(m = a + stats::runif(1)), list(a = 1), seed = 7)
stopifnot(is.null(ok$error) || is.na(ok$error), is.na(ok$warning), ok$value[["m"]] > 1)
again <- mc.run.one(function(a) c(m = a + stats::runif(1)), list(a = 1), seed = 7)
stopifnot(identical(ok$value, again$value))

bad <- mc.run.one(function() stop("boom"), list(), seed = 1)
stopifnot(is.null(bad$value), identical(bad$error, "boom"))
warned <- mc.run.one(function() {
  warning("careful")
  42
}, list(), seed = 1)
stopifnot(warned$value == 42, identical(warned$warning, "careful"))

stopifnot(all.equal(mc.bias(c(1, 2, 3, NA), 1), 1))
stopifnot(all.equal(mc.rmse(c(0, 2), 1), 1))
stopifnot(all.equal(mc.coverage(c(0, 2, NA), c(1, 3, 1), truth = 0.5), 0.5))
