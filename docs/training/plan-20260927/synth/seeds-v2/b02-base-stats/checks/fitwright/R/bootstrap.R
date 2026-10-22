set.seed(21)

b <- new_fw_boot(c(m = 1), matrix(1:3 + 0, ncol = 1), quote(f()))
stopifnot(inherits(b, "fw_boot"), b$R == 3L, identical(colnames(b$t), "m"))
stopifnot(inherits(try(new_fw_boot(c(1, 2), matrix(1, 2, 1), NULL), silent = TRUE), "try-error"))

idx <- boot_indices(6, 20, strata = rep(1:2, each = 3))
stopifnot(dim(idx) == c(20L, 6L), all(idx[, 1:3] <= 3), all(idx[, 4:6] >= 4))
stopifnot(all(boot_indices(5, 4) %in% 1:5))

x <- rnorm(100, 10)
bs <- boot_stat(x, function(v) c(mean = mean(v), sd = sd(v)), R = 200L)
stopifnot(dim(bs$t) == c(200L, 2L), identical(names(bs$t0), c("mean", "sd")))
bdf <- boot_stat(data.frame(v = x), function(d) mean(d$v), R = 50L)
stopifnot(identical(names(bdf$t0), "t1"), ncol(bdf$t) == 1L)

for (type in c("percentile", "basic", "normal")) {
  ci <- boot_ci(bs, type = type)
  stopifnot(identical(dimnames(ci), list(c("mean", "sd"), c("lower", "upper"))))
  stopifnot(all(ci[, "lower"] < bs$t0), all(ci[, "upper"] > bs$t0))
}
stopifnot(all.equal(boot_ci_basic(bs)[, "lower"], 2 * bs$t0 - apply(bs$t, 2, quantile, 0.975)))
stopifnot(inherits(try(boot_ci(list()), silent = TRUE), "try-error"))

d <- data.frame(x = runif(30), g = sample(c("a", "b"), 30, TRUE))
d$y <- 1 + 3 * d$x + rnorm(30, sd = 0.1)
bc <- boot_coefs(y ~ x, d, R = 40L)
stopifnot(identical(colnames(bc$t), c("(Intercept)", "x")), abs(mean(bc$t[, "x"]) - 3) < 0.3)
d$g[1] <- "rare"
bg <- boot_coefs(y ~ x + g, d, R = 60L)
stopifnot(ncol(bg$t) == 4L, all(is.na(bg$t[!stats::complete.cases(bg$t), ])))
