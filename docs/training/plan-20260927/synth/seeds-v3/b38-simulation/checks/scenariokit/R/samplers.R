err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
pn <- dist.param.names()
stopifnot(length(pn) == 6L, identical(pn$beta, c("shape1", "shape2")))

pars <- list(normal = list(mean = 2, sd = 0.5), lognormal = list(meanlog = 0, sdlog = 0.5),
  gamma = list(shape = 2, rate = 4), beta = list(shape1 = 2, shape2 = 3),
  uniform = list(min = 1, max = 3), poisson = list(lambda = 4))
stopifnot(identical(check.dist.params("normal", c(pars$normal, extra = 1)), pars$normal))
stopifnot(err(check.dist.params("gamma", list(shape = 1))) ==
  "The gamma distribution needs parameter `rate`.")
stopifnot(err(check.dist.params("normal", list(mean = Inf, sd = 1))) ==
  "Parameter `mean` of the normal distribution must be finite.")
stopifnot(err(check.dist.params("normal", list(mean = 0, sd = 0))) ==
  "Parameter `sd` of the normal distribution must be positive.")
stopifnot(err(check.dist.params("lognormal", list(meanlog = NA, sdlog = 1))) ==
  "Parameter `meanlog` of the lognormal distribution must be finite.")
stopifnot(err(check.dist.params("lognormal", list(meanlog = 0, sdlog = -1))) ==
  "Parameter `sdlog` of the lognormal distribution must be positive.")
stopifnot(err(check.dist.params("gamma", list(shape = 0, rate = 1))) ==
  "Parameter `shape` of the gamma distribution must be positive.")
stopifnot(err(check.dist.params("beta", list(shape1 = 1, shape2 = -1))) ==
  "Parameter `shape2` of the beta distribution must be positive.")
stopifnot(err(check.dist.params("uniform", list(min = 0, max = NaN))) ==
  "Parameter `max` of the uniform distribution must be finite.")
stopifnot(err(check.dist.params("poisson", list(lambda = 0))) ==
  "Parameter `lambda` of the poisson distribution must be positive.")

set.seed(123)
for (d in names(pars)) {
  x <- draw.sample(20000, d, pars[[d]])
  stopifnot(length(x) == 20000L)
  stopifnot(abs(mean(x) - dist.mean(d, pars[[d]])) < 0.05 * max(1, dist.mean(d, pars[[d]])))
  stopifnot(abs(var(x) / dist.var(d, pars[[d]]) - 1) < 0.1)
  fit <- fit.moments(x, d)
  stopifnot(identical(names(fit), pn[[d]]))
}
u <- draw.sample(100, "uniform", pars$uniform)
stopifnot(all(u >= 1 & u <= 3))
stopifnot(all(draw.sample(50, "poisson", pars$poisson) %% 1 == 0))

stopifnot(dist.mean("gamma", pars$gamma) == 0.5, dist.var("gamma", pars$gamma) == 0.125)
stopifnot(dist.mean("beta", pars$beta) == 0.4, isTRUE(all.equal(dist.var("beta",
  pars$beta), 0.04)))
stopifnot(dist.mean("uniform", pars$uniform) == 2, isTRUE(all.equal(dist.var("uniform",
  pars$uniform), 1 / 3)))
stopifnot(isTRUE(all.equal(dist.mean("lognormal", pars$lognormal), exp(0.125))))

stopifnot(isTRUE(all.equal(fit.moments(c(1, 3), "normal"), list(mean = 2, sd = sqrt(2)))))
stopifnot(identical(fit.moments(c(2, 4, NA), "poisson"), list(lambda = 3)))
g <- fit.moments(c(1, 3), "gamma")
stopifnot(g$shape == 2, g$rate == 1)
stopifnot(err(fit.moments(1, "normal")) == "`x` needs at least two non-missing values.")

