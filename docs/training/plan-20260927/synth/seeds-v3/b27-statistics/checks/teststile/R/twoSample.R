s = splitTwo(c(1, 2, NA, 4, 5), c("a", "a", "a", "b", "b"))
stopifnot(identical(s$x, c(1, 2)), identical(s$y, c(4, 5)))
stopifnot(inherits(try(splitTwo(1:3, c("a", "b", "c")), silent = TRUE), "try-error"))

set.seed(21)
x = stats::rnorm(15, 1)
y = stats::rnorm(20, 0, 2)

w = welchStat(x, y)
ref = stats::t.test(x, y)
stopifnot(all.equal(w$statistic, unname(ref$statistic)))
stopifnot(all.equal(w$parameter, unname(ref$parameter)))
stopifnot(all.equal(w$pValue, ref$p.value), w$method == "welch")

pl = pooledStat(x, y)
ref = stats::t.test(x, y, var.equal = TRUE)
stopifnot(all.equal(pl$statistic, unname(ref$statistic)), pl$parameter == 33)
stopifnot(all.equal(pl$pValue, ref$p.value))

rs = rankSumStat(c(x, 0.5), c(y, 0.5))
ref = stats::wilcox.test(c(x, 0.5), c(y, 0.5), exact = FALSE)
stopifnot(all.equal(rs$statistic, unname(ref$statistic)), all.equal(rs$pValue, ref$p.value))

k = ksStat(x, y)
ref = stats::ks.test(x, y, exact = FALSE)
stopifnot(all.equal(k$statistic, unname(ref$statistic)), abs(k$pValue - ref$p.value) < 1e-3)
stopifnot(ksStat(1:5, 1:5)$statistic == 0, ksStat(1:5, 1:5)$pValue == 1)

pv = c(0.01, 0.04, 0.03, NA, 0.2)
for (m in c("none", "bonferroni", "holm")) {
  stopifnot(all.equal(adjustPValues(pv, m), stats::p.adjust(pv, m)))
}
stopifnot(all.equal(adjustPValues(pv, "hochberg"), stats::p.adjust(pv, "hochberg")))
stopifnot(all.equal(adjustPValues(pv, "bh"), stats::p.adjust(pv, "BH")))
stopifnot(all.equal(adjustPValues(pv, "by"), stats::p.adjust(pv, "BY")))

df = data.frame(g = rep(c("u", "v"), c(15, 20)), a = c(x, y), b = c(x + 3, y))
for (tt in c("welch", "pooled", "rankSum", "ks")) {
  tab = twoSampleTable(df, c("a", "b"), "g", test = tt)
  stopifnot(identical(tab$variable, c("a", "b")), all(tab$nX == 15L))
  stopifnot(all(tab$pAdjusted >= tab$pValue))
}
tw = twoSampleTable(df, c("a", "b"), "g", adjust = "none")
stopifnot(all.equal(tw$pValue[1], stats::t.test(x, y)$p.value))
stopifnot(identical(tw$pValue, tw$pAdjusted))
stopifnot(inherits(try(twoSampleTable(df, "zz", "g"), silent = TRUE), "try-error"))
df$s = "q"
stopifnot(inherits(try(twoSampleTable(df, "s", "g"), silent = TRUE), "try-error"))
