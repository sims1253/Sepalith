set.seed(71)
dat <- data.frame(x = runif(40), g = factor(rep(c("a", "b"), 20)))
dat$y <- 3 * dat$x + ifelse(dat$g == "b", 2, 0) + rnorm(40, sd = 0.1)
lin <- stats::lm(y ~ x + g, dat)
quad <- stats::lm(y ~ x + I(x^2) + g, dat)

stopifnot(isTRUE(all.equal(predict_safely(lin, dat[1:3, ]), unname(predict(lin, dat[1:3, ])))))
stopifnot(inherits(try(predict_safely(lin, data.frame(z = 1)), silent = TRUE), "try-error"))

stopifnot(fd_step(c(-200, 1)) == 2e-3, fd_step(c(0, NA)) == 1e-5, fd_step(1, eps = 0.1) == 0.1)
stopifnot(fd_step(c(NA_real_, NA_real_)) == 1e-5)
stopifnot(inherits(try(fd_step(1, eps = -1), silent = TRUE), "try-error"))

sl <- slopes_fd(lin, dat, "x")
stopifnot(length(sl) == 40, isTRUE(all.equal(sl, rep(coef(lin)[["x"]], 40), tolerance = 1e-6)))
stopifnot(inherits(try(slopes_fd(lin, dat, "g"), silent = TRUE), "try-error"))

av <- average_slope(lin, dat, "x")
stopifnot(av$n == 40, abs(av$estimate - coef(lin)[["x"]]) < 1e-6, av$type == "response")
at <- slopes_at(quad, dat, "x", at = c(0, 1))
b <- coef(quad)
expected <- c(b[["x"]], b[["x"]] + 2 * b[["I(x^2)"]])
stopifnot(isTRUE(all.equal(at$estimate, expected, tolerance = 1e-5)))

by <- slopes_by(lin, dat, "x", by = "g")
stopifnot(identical(names(by)[1], "g"), identical(by$g, c("a", "b")), all(by$n == 20))

ct <- contrast_levels(lin, dat, "g")
stopifnot(ct$contrast == "b - a", isTRUE(all.equal(ct$estimate, coef(lin)[["gb"]])))
ct2 <- contrast_levels(lin, dat, "g", ref = "b")
stopifnot(isTRUE(all.equal(ct2$estimate, -coef(lin)[["gb"]])))
stopifnot(inherits(try(contrast_levels(lin, dat, "g", ref = "q"), silent = TRUE), "try-error"))
