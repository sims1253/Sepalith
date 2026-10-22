# formatMoney / formatPercent / formatCount
stopifnot(identical(formatMoney(c(1234.5, -20, NA)), c("$1,234.50", "-$20.00", "NA")))
stopifnot(identical(formatMoney(5, symbol = "EUR ", digits = 0), "EUR 5"))
stopifnot(identical(formatPercent(c(0.1234, NA)), c("12.3%", "NA")))
stopifnot(identical(formatCount(c(1234567, NA)), c("1,234,567", "NA")))

# padText
stopifnot(identical(padText("ab", 5), "ab   "), identical(padText("ab", 5, "right"), "   ab"))
stopifnot(identical(padText("ab", 5, "centre"), " ab  "))
stopifnot(identical(padText("abcdef", 3), "abcdef"))

# reportHeading / reportLine
h <- reportHeading("total", width = 11)
stopifnot(length(h) == 3, identical(h[2], "   TOTAL   "), identical(h[1], strrep("=", 11)))
stopifnot(identical(reportLine("Sales", "10", width = 12), "Sales ... 10"))
stopifnot(nchar(reportLine("A very long label", "x", width = 5)) > 5)

# reportTable
tab <- reportTable(data.frame(item = c("a", "bbb"), qty = c(5, 100)))
stopifnot(length(tab) == 4, identical(tab[1], "item  qty"), identical(tab[4], "bbb   100"))
stopifnot(identical(tab[2], strrep("-", 9)))
tab2 <- reportTable(data.frame(p = 0.5), formatters = list(p = formatPercent))
stopifnot(identical(tab2[3], "50.0%"))

# writeReport
tmp <- tempfile()
writeReport(c("one", "two"), tmp)
writeReport("three", tmp, append = TRUE)
stopifnot(identical(readLines(tmp), c("one", "two", "three")))
unlink(tmp)
