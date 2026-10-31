stopifnot(isTRUE(check_number_args(1, 2, ",", "")))
stopifnot(inherits(try(check_number_args("a", 2, ",", ""), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_number_args(1, -1, ",", ""), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_number_args(1, 2, 1, ""), silent = TRUE), "try-error"))
stopifnot(inherits(try(check_number_args(1, 2, ",", NA), silent = TRUE), "try-error"))

stopifnot(identical(fmt_fixed(c(1234.567, NA)), c("1,234.57", "")))
stopifnot(identical(fmt_fixed(0.5, digits = 0, na_string = "-"), "0"))
stopifnot(identical(fmt_fixed(NA_real_, na_string = "-"), "-"))

stopifnot(identical(fmt_percent(c(0.1234, NA)), c("12.3%", "")))
stopifnot(identical(fmt_percent(45, digits = 0, scale = 1), "45%"))

stopifnot(identical(fmt_integer(c(1234.6, NA)), c("1,235", "")))
stopifnot(identical(fmt_integer(1e6, big_mark = " "), "1 000 000"))

stopifnot(identical(number_style_spec("percent")$suffix, "%"))
stopifnot(number_style_spec("basis")$scale == 10000)
stopifnot(identical(number_style_spec("scientific", 3)$format, "e"))
stopifnot(number_style_spec("integer", 4)$digits == 0)
stopifnot(identical(number_style_spec()$format, "f"))

stopifnot(identical(fmt_number(1.234), "1.23"))
stopifnot(identical(fmt_number(0.25, "percent", 0), "25%"))
stopifnot(identical(fmt_number(0.0125, "basis", 0), "125 bp"))
stopifnot(identical(fmt_number(12345, "scientific", 1), "1.2e+04"))
stopifnot(identical(fmt_number(c(2.6, NA), "integer", na_string = "NA"), c("3", "NA")))

stopifnot(identical(fmt_signed(c(1.5, -2, 0, NA), digits = 1), c("+1.5", "-2.0", "0", NA)))
stopifnot(identical(fmt_signed(-1, digits = 0, minus = "(-)"), "(-)1"))

s <- summarise_numbers(c(1, 2, 3, NA, 2000), digits = 1)
stopifnot(identical(names(s), c("n", "mean", "sd", "min", "q1", "median", "q3", "max",
                               "missing")))
stopifnot(identical(unname(s["median"]), "2.5"), identical(unname(s["q3"]), "502.2"))
stopifnot(identical(unname(s[c("n", "missing", "max")]), c("4", "1", "2,000.0")))
stopifnot(identical(unname(s["min"]), "1.0"))
