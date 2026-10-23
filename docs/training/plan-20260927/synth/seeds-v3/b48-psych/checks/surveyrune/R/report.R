# skipped: needs ggplot2 (profile.plot)
stopifnot(identical(format.score(c(3.14159, NA)), c("3.1", "--")))
stopifnot(identical(format.score(c(-0.5, 1.25), "z", digits = 2), c("-0.50", "+1.25")))
stopifnot(identical(format.score(64.6, "t"), "T = 65"))
stopifnot(identical(format.score(99.4, "iq"), "IQ = 99"))
stopifnot(identical(format.score(c(0.2, 99.7, 50), "percentile"), c("PR = 1", "PR = 99",
  "PR = 50")))
stopifnot(identical(format.score(c(0, 12), "stanine"), c("Stanine 1", "Stanine 9")))
stopifnot(inherits(try(format.score("a"), silent = TRUE), "try-error"))

stopifnot(identical(band.label(c(25, 35, 50, 65, 75)),
  c("very low", "low", "average", "high", "very high")))
stopifnot(identical(band.label(c(39, 40, 60), "three"),
  c("below average", "average", "above average")))
stopifnot(identical(band.label(c(59, 62, 67, 70), "clinical"),
  c("normal", "elevated", "borderline", "clinical")))
stopifnot(identical(band.label(c(30, 50, 55), "strength"),
  c("development area", "typical", "strength")))

pr <- score.profile(c(anx = 14, dep = 6, slp = NA), means = c(anx = 10, dep = 8, slp = 5),
  sds = c(anx = 4, dep = 2, slp = 1))
stopifnot(identical(names(pr), c("scale", "raw", "z", "t", "percentile", "band")))
stopifnot(identical(pr$z[1:2], c(1, -1)), identical(pr$t[1:2], c(60, 40)))
stopifnot(identical(pr$percentile[1:2], c(84, 16)), identical(pr$band[1:2], c("high", "average")))
stopifnot(is.na(pr$band[3]))
stopifnot(inherits(try(score.profile(c(anx = 1), c(dep = 1), c(anx = 1)), silent = TRUE),
  "try-error"))

tx <- profile.text(pr, "Respondent 7")
stopifnot(length(tx) == 3)
stopifnot(identical(tx[1],
  "Respondent 7 scored in the high range on anx (T = 60, percentile 84)."))
stopifnot(identical(tx[3], "Respondent 7 has no valid score on slp."))
stopifnot(inherits(try(profile.text(pr[1:2]), silent = TRUE), "try-error"))

f <- tempfile(fileext = ".csv")
stopifnot(identical(export.profile(pr, f), f), nrow(read.csv(f)) == 3)
f2 <- tempfile(fileext = ".md")
export.profile(pr, f2, "markdown")
md <- readLines(f2)
stopifnot(length(md) == 5, startsWith(md[1], "| scale | raw"))
stopifnot(md[2] == "|---|---|---|---|---|---|")
f3 <- tempfile(fileext = ".html")
export.profile(pr, f3, "html")
ht <- readLines(f3)
stopifnot(ht[1] == "<table>", ht[6] == "</table>", startsWith(ht[2], "<tr><th>scale</th>"))
unlink(c(f, f2, f3))
