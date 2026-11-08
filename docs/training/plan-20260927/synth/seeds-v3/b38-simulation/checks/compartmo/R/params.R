err = function(expr) tryCatch({ expr; "" }, error = conditionMessage)
p = checkSeirParams(0.3, 0.2, 0.1)
stopifnot(identical(names(p), c("beta", "sigma", "gamma", "xi", "mu")), p$xi == 0)
stopifnot(err(checkSeirParams("a", 0.2, 0.1)) == "`beta` must be a single non-missing number.")
stopifnot(err(checkSeirParams(-1, 0.2, 0.1)) == "`beta` must be non-negative, not -1.")
stopifnot(err(checkSeirParams(1, NA, 0.1)) == "`sigma` must be a single non-missing number.")
stopifnot(err(checkSeirParams(1, -2, 0.1)) == "`sigma` must be non-negative, not -2.")
stopifnot(err(checkSeirParams(1, 1, c(1, 2))) == "`gamma` must be a single non-missing number.")
stopifnot(err(checkSeirParams(1, 1, -0.5)) == "`gamma` must be non-negative, not -0.5.")
stopifnot(err(checkSeirParams(1, 1, 1, xi = -1)) == "`xi` must be non-negative, not -1.")
stopifnot(err(checkSeirParams(1, 1, 1, mu = NULL)) == "`mu` must be a single non-missing number.")

d = defaultParams("sir")
stopifnot(d$model == "sir", identical(d$compartments, c("S", "I", "R")))
stopifnot(identical(names(defaultParams("seir")$params), c("beta", "sigma", "gamma", "mu")))
stopifnot(defaultParams("seirs")$params$xi == 0.01,
  length(defaultParams("sis")$compartments) == 2L)

q = paramsFromPeriods(2.5, latentPeriod = 4, infectiousPeriod = 5)
stopifnot(q$sigma == 0.25, q$gamma == 0.2, q$beta == 0.5, q$xi == 0)
stopifnot(paramsFromPeriods(2, 1, 1, immunityPeriod = 100)$xi == 0.01)
stopifnot(err(paramsFromPeriods(2, 0, 5)) == "`latentPeriod` must be a single positive number.")
stopifnot(err(paramsFromPeriods(2, 1,
  -5)) == "`infectiousPeriod` must be a single positive number.")
stopifnot(err(paramsFromPeriods(2, 1, 5,
  0)) == "`immunityPeriod` must be a single positive number.")
stopifnot(nchar(err(paramsFromPeriods(-1, 1, 5))) > 0)

w = scaleParams(list(beta = 0.3, gamma = 0.1, label = "x"))
stopifnot(isTRUE(all.equal(w$beta, 2.1)), isTRUE(all.equal(w$gamma, 0.7)), w$label == "x")
h = scaleParams(list(beta = 2.4), from = "day", to = "hour")
stopifnot(isTRUE(all.equal(h$beta, 0.1)))
stopifnot(isTRUE(all.equal(scaleParams(w, "week", "day")$beta, 0.3)))

ds = describeParams(list(beta = 0.3, sigma = 0.25, gamma = 0.1, xi = 0.01))
stopifnot(length(ds) == 4L, grepl("^R0: +3.0$", ds[1]), grepl("latent period: +4.0 days", ds[2]))
stopifnot(grepl("infectious period: +10.0 days", ds[3]), grepl("immunity: +100.0 days", ds[4]))
stopifnot(length(describeParams(list(beta = 0.3, gamma = 0.1, xi = 0))) == 2L)

stopifnot(isTRUE(all.equal(basicReproduction(list(beta = 0.3, gamma = 0.1)), 3)))
stopifnot(isTRUE(all.equal(basicReproduction(list(beta = 0.3, gamma = 0.1), "sis"), 3)))
pm = list(beta = 0.3, sigma = 0.2, gamma = 0.1, mu = 0.1)
stopifnot(isTRUE(all.equal(basicReproduction(pm, "seir"), 0.2 / 0.3 * 0.3 / 0.2)))
stopifnot(basicReproduction(pm, "seirs") == basicReproduction(pm, "seir"))
stopifnot(err(basicReproduction(list(beta = 1), "sis")) == "The sis model needs rate `gamma`.")
stopifnot(err(basicReproduction(list(beta = 1, gamma = 1), "seirs")) ==
  "The seirs model needs rate `sigma`.")
