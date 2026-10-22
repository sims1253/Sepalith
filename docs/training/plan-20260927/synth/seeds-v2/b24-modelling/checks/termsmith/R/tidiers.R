# check_formula() lives in formula-edit.R; minimal stand-in for this harness
check_formula <- function(x, arg = "formula") stopifnot(inherits(x, "formula"))

set.seed(61)
dat <- data.frame(x = rnorm(30), g = factor(rep(c("a", "b", "c"), 10)))
dat$y <- 1 + 2 * dat$x + rnorm(30, sd = 0.1)

fit0 <- fit_ridge(y ~ x + g, dat, lambda = 0)
ols <- stats::lm(y ~ x + g, dat)
stopifnot(inherits(fit0, "ts_ridge"), isTRUE(all.equal(fit0$coefficients, coef(ols))))
stopifnot(isTRUE(all.equal(fit0$edf, 4)), isTRUE(all.equal(fit0$sigma, summary(ols)$sigma)))
fit5 <- fit_ridge(y ~ x + g, dat, lambda = 5)
stopifnot(abs(fit5$coefficients[["x"]]) < abs(fit0$coefficients[["x"]]), fit5$edf < 4)
stopifnot(inherits(try(fit_ridge(y ~ x, dat, lambda = NA_real_), silent = TRUE), "try-error"))

td <- tidy.ts_ridge(fit0, conf.int = TRUE)
stopifnot(identical(td$term, names(coef(ols))), all(td$conf.low < td$estimate))
stopifnot(isTRUE(all.equal(td$std.error, unname(sqrt(diag(vcov(ols)))))))
gl <- glance.ts_ridge(fit0)
stopifnot(nrow(gl) == 1, isTRUE(all.equal(gl$r.squared, summary(ols)$r.squared)), gl$nobs == 30)

au <- augment.ts_ridge(fit0, data = dat)
stopifnot(isTRUE(all.equal(au$.fitted + au$.resid, dat$y)))
nd <- data.frame(x = c(0, NA, 1), g = factor(c("c", "a", "a"), levels = c("c", "a")))
pr <- augment.ts_ridge(fit0, newdata = nd)
stopifnot(nrow(pr) == 3, is.na(pr$.fitted[2]))
stopifnot(isTRUE(all.equal(pr$.fitted[c(1, 3)], unname(predict(ols, nd[c(1, 3), ])))))
stopifnot(inherits(try(augment.ts_ridge(fit0), silent = TRUE), "try-error"))

lr <- structure(list(statistic = 5, df = 1, p.value = 0.02, nobs = 30), class = "ts_lr_test")
stopifnot(tidy.ts_lr_test(lr)$method == "Likelihood-ratio test", glance.ts_lr_test(lr)$reject)
ic <- structure(data.frame(model = "m", ic = 1, logLik = -2), criterion = "BIC",
  class = c("ts_ic_table", "data.frame"))
ti <- tidy.ts_ic_table(ic)
stopifnot(identical(class(ti), "data.frame"), identical(names(ti), c("model", "bic", "loglik")))
