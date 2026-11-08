err <- function(expr) tryCatch({ expr; "" }, error = conditionMessage)
rg <- list(beta = c(0.1, 0.5), gamma = c(0.05, 0.25))
stopifnot(identical(check.ranges(rg), rg))
stopifnot(err(check.ranges(list(c(0, 1)))) == "`ranges` must be a named list.")
stopifnot(err(check.ranges(list(a = 1))) == "Range of `a` must be two non-missing numbers.")
stopifnot(err(check.ranges(list(a = c(2, 1)))) == "Range of `a` must be increasing.")

u <- data.frame(beta = c(0, 0.5, 1), gamma = c(1, 0.25, 0))
sc <- design.to.ranges(u, rg)
stopifnot(isTRUE(all.equal(sc$beta, c(0.1, 0.3, 0.5))), isTRUE(all.equal(sc$gamma,
  c(0.25, 0.1, 0.05))))
stopifnot(isTRUE(all.equal(design.to.unit(sc, rg), u)))

set.seed(4)
g <- design.points(3, rg, "grid")
stopifnot(nrow(g) == 9L, identical(names(g), c("beta", "gamma")),
  isTRUE(all.equal(sort(unique(g$beta)), c(0.1, 0.3, 0.5))))
r <- design.points(10, rg, "random")
stopifnot(nrow(r) == 10L, all(r$beta >= 0.1 & r$beta <= 0.5))
l <- design.points(8, rg, "lhs")
stopifnot(nrow(l) == 8L, identical(sort(floor(design.to.unit(l, rg)$gamma * 8)), 0:7 + 0))
o <- design.points(5, rg, "oat")
stopifnot(nrow(o) == 10L, isTRUE(all.equal(o$gamma[1:5], rep(0.15, 5))),
  isTRUE(all.equal(o$beta[6:10], rep(0.3, 5))))

fn <- function(p) 10 * p$beta - p$gamma
oe <- oat.effects(fn, rg, list(beta = 0.3, gamma = 0.1))
stopifnot(identical(oe$parameter, c("beta", "gamma")), isTRUE(all.equal(oe$effect, c(4, -0.2))))
stopifnot(err(oat.effects(fn, rg, list(beta = 1))) == "`base` has no value for `gamma`.")
stopifnot(isTRUE(all.equal(oe$low, c(0.9, 2.95))), isTRUE(all.equal(oe$high, c(4.9, 2.75))))

set.seed(5)
d <- design.points(50, rg, "lhs")
y <- 10 * d$beta - d$gamma + rnorm(50, sd = 0.01)
for (m in c("prcc", "pearson", "spearman")) {
  sc <- sensitivity.coef(d, y, m)
  stopifnot(identical(names(sc), c("beta", "gamma")), sc[["beta"]] > 0.9, sc[["gamma"]] < 0)
}
stopifnot(nchar(err(sensitivity.coef(d, y[-1]))) > 0)
