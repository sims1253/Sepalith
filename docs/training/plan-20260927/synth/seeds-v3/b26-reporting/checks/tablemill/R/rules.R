stopifnot(identical(normaliseAlign("lrc", 3), c("left", "right", "centre")))
stopifnot(identical(normaliseAlign(c("left", "center"), 4), rep(c("left", "centre"), 2)))
stopifnot(inherits(try(normaliseAlign("middle", 2), silent = TRUE), "try-error"))
stopifnot(inherits(try(normaliseAlign("lr", 3), silent = TRUE), "try-error"))

stopifnot(identical(borderChars()$top, "+"))
stopifnot(identical(borderChars("rounded")$bottom, "'"))
stopifnot(identical(borderChars("double")$horizontal, "="))
stopifnot(identical(borderChars("minimal")$vertical, " "))
stopifnot(identical(borderChars("heavy")$middle, "#"))

stopifnot(identical(boxRule(c(1, 2)), "+---+----+"))
stopifnot(identical(boxRule(2, "bottom", "rounded", padding = 0L), "'--'"))
stopifnot(identical(boxRule(1, "middle", "double"), "+===+"))

stopifnot(identical(boxRow(c("a", "b"), c(2, 3), "lr"), "| a  |   b |"))
stopifnot(identical(boxRow("x", 1, style = "heavy", padding = 0L), "#x#"))

bt = boxTable(matrix(c("1", "22"), ncol = 1), "n", align = "right")
stopifnot(identical(bt, c("+----+", "|  n |", "+----+", "|  1 |", "| 22 |", "+----+")))
stopifnot(inherits(try(boxTable(matrix("a"), c("x", "y")), silent = TRUE), "try-error"))

stopifnot(identical(markdownRule(c(3, 4, 5), "lrc"), "| :-- | ---: | :---: |"))
stopifnot(identical(markdownRule(2), "| :-- |"))

stopifnot(identical(latexColSpec("lrr"), "lrr"))
stopifnot(identical(latexColSpec(c("left", "centre"), vertical = TRUE), "|l|c|"))
stopifnot(identical(latexColSpec("right", n = 2), "rr"))

stopifnot(identical(latexRule(), "\\toprule"))
stopifnot(identical(latexRule("mid"), "\\midrule"))
stopifnot(identical(latexRule("bottom", booktabs = FALSE), "\\hline"))

tt = boxTable(matrix("a"), "x", title = "T")
stopifnot(identical(tt[1], "  T  "), length(tt) == 6L, identical(tt[2], "+---+"))
stopifnot(inherits(try(boxTable(matrix("a"), "x", title = 1), silent = TRUE), "try-error"))
stopifnot(identical(boxTable(data.frame(v = "a"), "x"), tt[-1]))
