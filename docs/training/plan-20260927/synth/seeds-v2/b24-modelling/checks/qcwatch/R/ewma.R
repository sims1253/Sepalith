stopifnot(isTRUE(all.equal(ewma.stat(c(1, NA, 3), lambda = 0.5, start = 0), c(0.5, 0.5, 1.75))))
stopifnot(identical(ewma.stat(c(2, 4), lambda = 1), c(2, 4)))
stopifnot(inherits(try(ewma.stat(1, lambda = 0), silent = TRUE), "try-error"))

lim <- ewma.limits(3, target = 10, sigma = 2, lambda = 1, L = 3)
stopifnot(identical(lim$ucl, rep(16, 3)), identical(lim$lcl, rep(4, 3)))
grow <- ewma.limits(50, 0, 1, lambda = 0.2)
stopifnot(all(diff(grow$ucl) > 0))
stopifnot(isTRUE(all.equal(ewma.limits(1, 0, 1, asymptotic = TRUE)$ucl, 1)))

ch <- ewma.chart(c(rep(0, 5), rep(5, 5)), target = 0, sigma = 1, lambda = 0.5)
stopifnot(inherits(ch, "qc_ewma"), ch$run.length == 6L, all(6:10 %in% ch$signals))
stopifnot(inherits(try(ewma.chart(1:3, 0, sigma = -1), silent = TRUE), "try-error"))

set.seed(51)
arl <- suppressWarnings(ewma.arl.sim(lambda = 0.5, L = 2, delta = 3, n.sim = 50, max.run = 100))
stopifnot(arl$arl < 3, arl$censored == 0, arl$se >= 0)

cmp <- ewma.compare(c(rep(0, 5), rep(5, 5)), target = 0, sigma = 1, lambdas = c(0.2, 1))
stopifnot(identical(cmp$lambda, c(0.2, 1)), cmp$first.signal[2] == 6L, all(cmp$n.signals > 0))

out <- capture.output(pr <- print(ch))
stopifnot(identical(pr, ch), grepl("lambda = 0.5", out[2]))
sm <- summary(ch)
stopifnot(!sm$in.control, sm$max.excursion > 1, identical(sm$signals, ch$signals))
