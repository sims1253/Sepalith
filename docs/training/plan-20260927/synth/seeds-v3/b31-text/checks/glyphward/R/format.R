stopifnot(identical(fmt.bytes(c(512, 1536, 1048576 * 2.25, NA)),
  c("512 B", "1.5 KiB", "2.2 MiB", NA)))
stopifnot(identical(fmt.bytes(1024^5, digits = 0L), "1024 TiB"))
stopifnot(identical(fmt.bytes(1500, si = TRUE), "1.5 kB"))

stopifnot(identical(fmt.duration(c(3723, 245, 7, NA)), c("1h 02m 03s", "4m 05s", "7s", NA)))
stopifnot(identical(fmt.duration(3600), "1h 00m 00s"))
stopifnot(identical(fmt.duration(59.6), "1m 00s"))

stopifnot(identical(fmt.value(2048, "bytes"), "2.0 KiB"))
stopifnot(identical(fmt.value(65, "duration", width = 8L), "  1m 05s"))
stopifnot(identical(fmt.value(c(0.125, NA), "percent"), c("12.5%", "-")))
stopifnot(identical(fmt.value(1/3, "percent", digits = 0L), "33%"))
stopifnot(identical(fmt.value(c(1234567, 999.6), "count"), c("1,234,567", "1,000")))
stopifnot(identical(fmt.value(c(1, 2, 3, 4, 11, 12, 13, 21, 112, NA), "ordinal"),
  c("1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "112th", "-")))

d <- describe.numbers(c(1, 2, 3, 4, NA))
stopifnot(identical(names(d), c("n", "mean", "sd", "min", "q1", "median", "q3", "max")))
stopifnot(identical(unname(d[c("n", "mean", "median")]), c("4", "2.50", "2.50")))
stopifnot(identical(attr(describe.numbers(c(1, 3), digits = 0L), "line"),
  "n=2 mean=2 sd=1 min=1 q1=2 median=2 q3=2 max=3"))

tb <- fmt.table(data.frame(name = c("a", "bbb"), n = c(1, 22)), align = c("l", "r"))
stopifnot(identical(tb, c("name   n", "----  --", "a      1", "bbb   22", "----  --")))
tb2 <- fmt.table(data.frame(x = c(NA, "yy")), sep = "|", title = "T", footer = "F")
stopifnot(identical(tb2, c("T", "x ", "--", "  ", "yy", "--", "F")))
stopifnot(identical(fmt.table(data.frame(a = character())), c("a", "-", "-")))
stopifnot(inherits(tryCatch(fmt.table(data.frame(a = 1), align = "c"), error = identity),
  "error"))

kv <- fmt.kv(list(name = "x", tags = c("a", "b"), none = NULL), indent = 2L)
stopifnot(identical(kv, c("  name: x", "  tags: a, b", "  none: ")))
stopifnot(identical(fmt.kv(c(a = 1, bb = 2), sep = " = "), c("a  = 1", "bb = 2")))
stopifnot(inherits(tryCatch(fmt.kv(list(1)), error = identity), "error"))
