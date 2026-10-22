set.seed(11)
d <- data.frame(x = runif(40), z = rnorm(40))
d$y <- 1 + 2 * d$x - 0.5 * d$z + rnorm(40, sd = 0.2)

r1 <- fit_lm_safely(y ~ ., d)
stopifnot(r1$ok, is.na(r1$message), identical(names(coef(r1$fit)), c("(Intercept)", "x", "z")))
r2 <- fit_lm_safely(y ~ x, d, weights = rep(2, 40))
stopifnot(r2$ok, all.equal(coef(r2$fit), coef(lm(y ~ x, d))))
d$x2 <- 2 * d$x
r3 <- fit_lm_safely(y ~ x + x2, d)
stopifnot(!r3$ok, grepl("x2", r3$message), !is.null(r3$fit))
stopifnot(!fit_lm_safely(y ~ nope, d)$ok)
stopifnot(inherits(try(fit_lm_safely(y ~ x, d, weights = 1:3), silent = TRUE), "try-error"))

fit <- lm(y ~ x + z, d)
ct <- coef_table_lm(fit)
stopifnot(identical(ct$term, c("(Intercept)", "x", "z")))
stopifnot(all.equal(ct$p_value, unname(summary(fit)$coefficients[, 4])))
stopifnot(all.equal(as.matrix(ct[c("lower", "upper")]), unname(confint(fit)),
  check.attributes = FALSE))

d$b <- rbinom(40, 1, plogis(d$x - 0.5))
gfit <- glm(b ~ x, binomial, d)
gt <- coef_table_glm(gfit)
stopifnot(all.equal(gt$p_value, unname(summary(gfit)$coefficients[, 4])))
stopifnot(all.equal(coef_table_glm(gfit, exponentiate = TRUE)$estimate, exp(gt$estimate)))

stopifnot(all.equal(model_residuals(fit, "standardized"), rstandard(fit)))
stopifnot(all.equal(model_residuals(fit, "studentized"), rstudent(fit)))
stopifnot(identical(model_residuals(fit), residuals(fit)))

stopifnot(identical(leverage_flags(fit), hatvalues(fit) > 2 * mean(hatvalues(fit))))
stopifnot(!any(leverage_flags(fit, threshold = 1)))
stopifnot(identical(cooks_flags(fit), cooks.distance(fit) > 0.1))
stopifnot(identical(dffits_flags(fit), abs(dffits(fit)) > 2 * sqrt(3 / 40)))

small <- lm(y ~ x, d)
cn <- compare_nested(small, fit)
stopifnot(all.equal(cn$p_value, anova(small, fit)[2, "Pr(>F)"]), cn$df == 1)
stopifnot(compare_nested(small, fit, "Chisq")$statistic > 0)
stopifnot(inherits(try(compare_nested(fit, small), silent = TRUE), "try-error"))

cv <- cv_lm(y ~ x + z, d, k = 5L, seed = 1)
stopifnot(nrow(cv$folds) == 5L, !anyNA(cv$predictions), cv$rmse >= cv$mae)
stopifnot(identical(cv_lm(y ~ x, d, k = 4L, seed = 2)$predictions,
  cv_lm(y ~ x, d, k = 4L, seed = 2)$predictions))
stopifnot(inherits(try(cv_lm(y ~ x, d, k = 1L), silent = TRUE), "try-error"))
