# register the S3 methods as the package NAMESPACE would
registerS3method("format", "ts_ic_table", format.ts_ic_table)
registerS3method("format", "ts_lr_test", format.ts_lr_test)

set.seed(81)
dat <- data.frame(x = rnorm(25), z = rnorm(25))
dat$y <- 1 + dat$x + rnorm(25)
m0 <- stats::lm(y ~ 1, dat)
m1 <- stats::lm(y ~ x, dat)
m2 <- stats::lm(y ~ x + z, dat)

k <- attr(logLik(m1), "df")
stopifnot(isTRUE(all.equal(aicc(m1), AIC(m1) + 2 * k * (k + 1) / (25 - k - 1))))
tiny <- stats::lm(y ~ x, dat[1:3, ])
stopifnot(isTRUE(all.equal(suppressWarnings(aicc(tiny)), AIC(tiny))))

w <- akaike_weights(c(10, 12, 10))
stopifnot(isTRUE(all.equal(sum(w), 1)), w[1] == w[3], w[2] < w[1])
stopifnot(inherits(try(akaike_weights(c(1, NA)), silent = TRUE), "try-error"))

tab <- ic_table(null = m0, x = m1, xz = m2)
stopifnot(inherits(tab, "ts_ic_table"), tab$model[1] == "x", tab$delta[1] == 0)
stopifnot(identical(sort(tab$model), c("null", "x", "xz")), attr(tab, "criterion") == "AIC")
stopifnot(identical(ic_table(m0, m1, criterion = "BIC")$model[1], "model_2"))
stopifnot(inherits(try(ic_table(m0, m1, .names = "a"), silent = TRUE), "try-error"))

lr <- lr_test(m0, m1)
stopifnot(inherits(lr, "ts_lr_test"), lr$df == 1, lr$p.value < 0.01, lr$nobs == 25)
stopifnot(inherits(try(lr_test(m1, m0), silent = TRUE), "try-error"))
seq_tab <- lr_sequence(list(m0, m1, m2))
stopifnot(nrow(seq_tab) == 2, all(seq_tab$p.adjusted >= seq_tab$p.value))

ft <- format(tab)
stopifnot(identical(class(ft), "data.frame"), "AIC" %in% names(ft), is.character(ft$weight))
out <- capture.output(res <- print(tab))
stopifnot(identical(res, tab), identical(out[1], "<model comparison by AIC>"))
stopifnot(grepl("^LR = [0-9.]+, df = 1, p = ", format(lr)))
stopifnot(identical(capture.output(print(lr))[1], "<likelihood-ratio test>"))
