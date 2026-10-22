# register the S3 method as the package NAMESPACE would
registerS3method("format", "pcSmooth", format.pcSmooth)
s1 <- smoothTerm("age", bs = "cr", k = 10)
stopifnot(inherits(s1, "pcSmooth"), identical(format(s1), "s(age, bs = \"cr\", k = 10)"))
re <- suppressWarnings(smoothTerm("site", bs = "re", k = 5))
stopifnot(re$k == -1, identical(format(re), "s(site, bs = \"re\")"))
stopifnot(inherits(try(smoothTerm(c("a", "b")), silent = TRUE), "try-error"))

te <- tensorTerm(c("x", "z"), k = 4)
stopifnot(identical(te$k, c(4, 4)), identical(format(te), "te(x, z, bs = \"cr\", k = c(4, 4))"))
ti <- tensorTerm(c("x", "z"), interactionOnly = TRUE)
stopifnot(identical(format(ti), "ti(x, z, bs = \"cr\")"))
stopifnot(inherits(try(tensorTerm("x"), silent = TRUE), "try-error"))

byS <- smoothTerm("age", by = "sex")
f <- gamFormula("y", list(byS, te), parametric = "trt")
stopifnot(identical(deparse1(f),
  "y ~ trt + sex + s(age, bs = \"tp\", by = sex) + te(x, z, bs = \"cr\", k = c(4, 4))"))
stopifnot(identical(deparse1(gamFormula("y", list())), "y ~ 1"))

dat <- data.frame(age = rep(1:6, 5), x = rep(1:3, 10), z = 1:30, site = gl(3, 10))
bs <- checkBasisSize(list(s1, te, re), dat)
stopifnot(nrow(bs) == 3)
stopifnot(identical(bs$ok, c(FALSE, FALSE, TRUE)))
stopifnot(identical(bs$unique, c(6L, 3L, 30L)))
stopifnot(inherits(try(checkBasisSize(smoothTerm("w"), dat), silent = TRUE), "try-error"))

# mgcv is a recommended package, so the fitting helpers can run here too
set.seed(41)
dat$y <- sin(dat$z / 5) + rnorm(30, sd = 0.1)
sms <- list(smoothTerm("z", k = 8), smoothTerm("age", k = 10))
fit <- suppressWarnings(fitGamSpec("y", sms, dat))
stopifnot(inherits(fit, "gam"), length(fit$smooth) == 2)
ss <- smoothSummary(fit)
stopifnot(identical(ss$term, c("s(z)", "s(age)")), ss$wiggly[1], all(ss$p.value >= 0))
