rho <- autocorrAtLags(c(1, 3, 2, 5), maxLag = 10)
stopifnot(identical(names(rho), c("1", "2", "3")), all.equal(rho[["1"]], -2.3125 / 8.75))

set.seed(61)
stopifnot(all(abs(autocorrAtLags(rnorm(4000), 5)) < 0.06))
stopifnot(effectiveSize(rep(2, 10)) == 0)
ess <- effectiveSize(rnorm(2000))
stopifnot(ess > 1500, ess <= 2000)
ar1 <- as.numeric(filter(rnorm(3000), 0.9, method = "recursive"))
stopifnot(effectiveSize(ar1) < 400)

ch <- matrix(1:20, ncol = 2)
stopifnot(identical(burnAndThin(ch, 2, 3)[, 1], c(3L, 6L, 9L)))
stopifnot(nrow(burnAndThin(1:5)) == 5L)
stopifnot(inherits(try(burnAndThin(ch, 10), silent = TRUE), "try-error"))

stopifnot(abs(gelmanRubin(replicate(3, rnorm(1000), simplify = FALSE)) - 1) < 0.02)
stopifnot(gelmanRubin(list(rnorm(300), rnorm(300, 4))) > 2)
stopifnot(inherits(try(gelmanRubin(list(1:3, 1:4)), silent = TRUE), "try-error"))

stopifnot(abs(gewekeZ(rnorm(2000))) < 3, gewekeZ(c(rnorm(500, 5), rnorm(1500))) > 3)
stopifnot(inherits(try(gewekeZ(rnorm(10), 0.6, 0.6), silent = TRUE), "try-error"))

s <- summarizeChain(cbind(a = rnorm(200), b = rexp(200)))
stopifnot(inherits(s, "chainletSummary"), identical(s$param, c("a", "b")))
stopifnot(identical(names(s), c("param", "mean", "sd", "ess", "2.5%", "50%", "97.5%")))
s2 <- summarizeChain(matrix(rnorm(100), ncol = 2))
stopifnot(identical(s2$param, c("par1", "par2")))

out <- utils::capture.output(res <- print(s))
stopifnot(identical(res, s), out[1] == "Chain summary: 2 parameter(s)", length(out) == 4L)
