# skipped: projectionUI, projectionServer (need shiny / bslib);
#   plotProjection (needs ggplot2)

# checkAmount / checkRate / checkMonths
stopifnot(is.null(checkAmount(0, "Start")))
stopifnot(identical(checkAmount(NA, "Start"), "Start is required."))
stopifnot(identical(checkAmount(-1, "Start"), "Start cannot be negative."))
stopifnot(is.null(checkRate(2.5)), grepl("between", checkRate(30)))
stopifnot(grepl("required", checkRate(NULL)))
stopifnot(is.null(checkMonths(12)), grepl("whole number", checkMonths(1.5)))
stopifnot(grepl("whole number", checkMonths(0)))

# projectBalance
p <- projectBalance(1000, 100, 0, 3)
stopifnot(identical(p$balance, c(1100, 1200, 1300)), identical(p$contributed, c(1100, 1200, 1300)))
q <- projectBalance(1200, 0, 0.12, 2)
stopifnot(isTRUE(all.equal(q$interest, c(12, 24.12))), isTRUE(all.equal(q$balance[2], 1224.12)))

# summarizeProjection works for plans shorter than a year
s <- summarizeProjection(projectBalance(1000, 100, 0.02, 5))
stopifnot(identical(s$Month, 5L), identical(names(s)[3], "Interest earned"))
stopifnot(identical(summarizeProjection(projectBalance(0, 1, 0, 30))$Month, c(12L, 24L, 30L)))
stopifnot(identical(summarizeProjection(projectBalance(0, 1, 0, 24))$Month, c(12L, 24L)))
