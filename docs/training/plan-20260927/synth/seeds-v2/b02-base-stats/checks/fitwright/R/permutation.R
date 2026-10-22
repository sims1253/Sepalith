set.seed(31)

p <- new_fw_perm(2, c(-3, -1, 0, 1, 2.5), "two.sided", "demo")
stopifnot(inherits(p, "fw_perm"), p$n_perm == 5L, p$p_value == 3 / 6)

stopifnot(perm_p_value(2, c(-3, -1, 0, 1, 2.5)) == 3 / 6)
stopifnot(perm_p_value(2, c(-3, -1, 0, 1, 2.5), "greater") == 2 / 6)
stopifnot(perm_p_value(2, c(-3, -1, 0, 1, 2.5), "less") == 5 / 6)

x <- rnorm(20, 1)
y <- rnorm(25)
pm <- perm_test_mean(c(x, NA), y, n_perm = 499L)
stopifnot(all.equal(pm$statistic, mean(x) - mean(y)), pm$p_value < 0.05, pm$n_perm == 499L)
stopifnot(perm_test_mean(y[1:10], y[11:20], 199L, "greater")$p_value > 0.01)

pmed <- perm_test_median(x, y, n_perm = 199L)
stopifnot(all.equal(pmed$statistic, median(x) - median(y)), pmed$p_value <= 1)

u <- runif(30)
pc <- perm_test_cor(u, u + rnorm(30, sd = 0.1), n_perm = 199L, method = "spearman")
stopifnot(pc$statistic > 0.8, pc$p_value == 1 / 200, grepl("spearman", pc$method))

pp <- perm_test_paired(x[1:15] + 2, x[1:15] + rnorm(15, sd = 0.1), n_perm = 299L)
stopifnot(pp$p_value < 0.01, length(pp$null_dist) == 299L)
stopifnot(inherits(try(perm_test_paired(1:3, 1:4), silent = TRUE), "try-error"))

yy <- c(rnorm(10), rnorm(10, 3), NA)
gg <- factor(c(rep("a", 10), rep("b", 10), "a"), levels = c("a", "b", "unused"))
po <- perm_test_oneway(yy, gg, n_perm = 199L)
ref_f <- anova(lm(yy[1:20] ~ gg[1:20]))[1, "F value"]
stopifnot(all.equal(po$statistic, ref_f), po$alternative == "greater", po$p_value < 0.05)

d <- data.frame(a = rnorm(40), b = rnorm(40))
d$y <- d$a + rnorm(40)
pl <- perm_test_lm_term(y ~ a + b, d, "a", n_perm = 199L)
stopifnot(all.equal(pl$statistic, summary(lm(y ~ a + b, d))$coefficients["a", 3]))
stopifnot(pl$p_value < 0.05, grepl("`a`", pl$method))
stopifnot(inherits(try(perm_test_lm_term(y ~ a, d, "b"), silent = TRUE), "try-error"))
