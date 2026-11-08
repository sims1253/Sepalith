cut = diagnosticCutoffs(50, 5)
stopifnot(cut[["leverage"]] == 0.2, cut[["cooks"]] == 0.08)
stopifnot(all.equal(cut[["dffits"]], 2 * sqrt(0.1)), cut[["rstudent"]] == 2)

set.seed(41)
d = data.frame(x1 = stats::rnorm(40), x2 = stats::rnorm(40))
d$x3 = d$x1 + stats::rnorm(40, sd = 0.3)
d$y = 1 + d$x1 - d$x2 + stats::rnorm(40)
d$y[5] = d$y[5] + 8
fit = stats::lm(y ~ x1 + x2 + x3, data = d)

im = influenceMeasures(fit)
stopifnot(identical(names(im), c("leverage", "rstandard", "rstudent", "cooks", "dffits")))
stopifnot(all.equal(im$leverage, unname(stats::hatvalues(fit))))
stopifnot(all.equal(im$rstandard, unname(stats::rstandard(fit))))
stopifnot(all.equal(im$rstudent, unname(stats::rstudent(fit))))
stopifnot(all.equal(im$cooks, unname(stats::cooks.distance(fit))))
stopifnot(all.equal(im$dffits, unname(stats::dffits(fit))))
im2 = influenceMeasures(fit, c("cooks", "leverage"))
stopifnot(identical(names(im2), c("cooks", "leverage")))

fl = flagInfluential(fit)
stopifnot(identical(colnames(fl), c("leverage", "rstudent", "cooks", "dffits")))
stopifnot(fl[5, "rstudent"])
stopifnot(attr(fl, "cutoffs")[["cooks"]] == 0.1, nrow(fl) == 40L)

stopifnot(all.equal(residualsByType(fit), unname(stats::residuals(fit))))
stopifnot(all.equal(residualsByType(fit, "scaled"),
  unname(stats::residuals(fit)) / stats::sigma(fit)))
stopifnot(all.equal(residualsByType(fit, "standardized"), im$rstandard))
stopifnot(all.equal(residualsByType(fit, "studentized"), im$rstudent))

vf = vifValues(fit)
stopifnot(identical(names(vf), c("x1", "x2", "x3")), vf[["x1"]] > 5, vf[["x2"]] < 2)
stopifnot(inherits(try(vifValues(stats::lm(y ~ x1, data = d)), silent = TRUE), "try-error"))

bp = breuschPagan(fit)
stopifnot(bp$parameter == 3, bp$statistic >= 0, bp$pValue > 0, bp$pValue <= 1)

dw = durbinWatson(fit)
stopifnot(dw > 0, dw < 4)
stopifnot(durbinWatson(stats::lm(c(1, -1, 1, -1, 1, -1) ~ 1)) > 3)
stopifnot(inherits(try(influenceMeasures(list()), silent = TRUE), "try-error"))
stopifnot(inherits(try(durbinWatson(stats::lm(c(1, 2) ~ 1)), silent = TRUE), "try-error"))

dr = diagnosticReport(fit)
stopifnot(identical(dr$check, c("leverage", "outliers", "influence", "heteroscedasticity",
  "autocorrelation", "collinearity")))
stopifnot(dr$flagged[2], all.equal(dr$value[2], max(abs(stats::rstudent(fit)))))
stopifnot(dr$threshold[6] == 10)
stopifnot(all.equal(dr$value[4], bp$pValue), dr$flagged[4] == (bp$pValue < 0.05))
stopifnot(all.equal(dr$value[6], max(vf)), dr$flagged[6] == (max(vf) > 10))
dr1 = diagnosticReport(stats::lm(y ~ x1, data = d))
stopifnot(is.na(dr1$value[6]), is.na(dr1$flagged[6]), nrow(dr1) == 6L)
