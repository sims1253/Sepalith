# sys.source() turns parse data off while it evaluates this file
options(keep.parse.data = TRUE)

src <- c(
  "#' Doc",
  "f <- function(x) {",
  "",
  "  # note",
  "  if (x && TRUE) { y <- \"a",
  "b\" } # tail",
  "  for (i in 1) g(i)",
  "}"
)

fm <- fileMetrics(src)
stopifnot(identical(names(fm), c("lines", "blank", "comment", "code", "maxWidth", "meanWidth",
  "functions", "assignments")))
stopifnot(fm$lines == 8L, fm$blank == 1L, fm$comment == 2L, fm$code == 5L)
stopifnot(fm$maxWidth == 26L, abs(fm$meanWidth - 89 / 7) < 1e-12)
stopifnot(fm$functions == 1L, fm$assignments == 2L)
fe <- fileMetrics(character())
stopifnot(fe$lines == 0L, fe$maxWidth == 0L, fe$meanWidth == 0)

stopifnot(metricValue(src, "sloc") == 5L, metricValue(src, "commentRatio") == 3 / 8)
stopifnot(metricValue(src, "cyclomatic") == 4L, metricValue(src, "maxNesting") == 2L)
stopifnot(abs(metricValue(src, "halsteadVolume") - 123.1899) < 1e-4)
stopifnot(metricValue(src, "uniqueCalls") == 1L)
stopifnot(metricValue("g(1); g(2); h()", "uniqueCalls") == 2L)
stopifnot(metricValue("", "commentRatio") == 0, metricValue("", "halsteadVolume") == 0)
stopifnot(metricValue("", "maxNesting") == 0L, metricValue("", "cyclomatic") == 1L)

lc <- lineClasses(src)
stopifnot(identical(lc, c("roxygen", "code", "blank", "comment", "code", "mixed", "code", "code")))
lc2 <- lineClasses(c("x <- '", "  text", "'", "", "1 # one"))
stopifnot(identical(lc2, c("code", "continued", "continued", "blank", "mixed")))

cm <- compareMetrics(src, src[-4])
stopifnot(nrow(cm) == 6L, identical(cm$metric[1], "sloc"), all(cm$change[-2] == 0))
stopifnot(abs(cm$after[2] - 2 / 7) < 1e-12, cm$change[2] < 0)
cm2 <- compareMetrics("x", "if (a) b", metrics = c("cyclomatic", "uniqueCalls"))
stopifnot(identical(cm2$change, c(1, 0)), identical(cm2$relative, c(1, NA)))

th <- checkThresholds(src, c(sloc = 4, cyclomatic = 30, maxNesting = 1, commentRatio = 0.5))
stopifnot(identical(th$metric, c("sloc", "cyclomatic", "maxNesting", "commentRatio")))
stopifnot(identical(th$ok, c(FALSE, TRUE, FALSE, FALSE)), th$message[2] == "")
stopifnot(identical(th$message[1], "sloc is 5, above the limit of 4."))
stopifnot(identical(th$message[3], "maxNesting is 2, above the limit of 1."))
stopifnot(identical(th$message[4], "commentRatio is 0.375, below the limit of 0.5."))
stopifnot(all(checkThresholds(src)$ok == c(TRUE, TRUE, TRUE, TRUE)))
