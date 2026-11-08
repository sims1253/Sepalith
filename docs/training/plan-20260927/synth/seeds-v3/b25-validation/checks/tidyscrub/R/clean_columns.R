stopifnot("N/A" %in% default.na.tokens(), "" %in% default.na.tokens())

stopifnot(identical(scrub(c(1, -99, NaN, Inf, 9999, 2)), c(1, NA, NA, NA, NA, 2)))
stopifnot(identical(scrub(c(1, -99), na.values = numeric()), c(1, -99)))
stopifnot(identical(scrub(c("  a   b ", "N/A", "", "x")), c("a b", NA, NA, "x")))
stopifnot(identical(scrub(c(" a ", "?"), squish = FALSE), c(" a ", NA)))
f <- scrub(factor(c(" lo", "lo", "hi ", "n/a")))
stopifnot(identical(levels(f), c("lo", "hi")), identical(as.character(f), c("lo", "lo", "hi", NA)))
stopifnot(identical(scrub(c(TRUE, NA)), c(TRUE, NA)))
stopifnot(identical(scrub(as.Date("2024-01-01")), as.Date("2024-01-01")))

n <- coerce.column(c("1,234.5", "x", NA))
stopifnot(identical(as.vector(n), c(1234.5, NA, NA)), identical(attr(n, "failed"), 2L))
stopifnot(identical(as.vector(coerce.column("1.234,5", dec = ",")), 1234.5))
i <- coerce.column(c("3", "3.5", "1,000"), "integer")
stopifnot(identical(as.vector(i), c(3L, NA, 1000L)), identical(attr(i, "failed"), 2L))
l <- coerce.column(c("Yes", "n", "0", "maybe"), "logical")
stopifnot(identical(as.vector(l), c(TRUE, FALSE, FALSE, NA)), identical(attr(l, "failed"), 4L))
d <- coerce.column(c("2024-01-02", "02.01.2024"), "date")
stopifnot(identical(as.character(d), c("2024-01-02", "2024-01-02")))
stopifnot(identical(attr(coerce.column(c("a", NA), "character"), "failed"), integer()))

raw <- data.frame(
  id = c(" 1 ", "2", "", "4"),
  amount = c("10", "n/a", "", "x"),
  score = c(1, -99, NA, 3),
  stringsAsFactors = FALSE
)
res <- clean.frame(raw, types = c(id = "integer", amount = "numeric"))
stopifnot(identical(names(res), c("data", "changed", "failed", "dropped")))
stopifnot(identical(res$dropped, 3L), nrow(res$data) == 3L)
stopifnot(identical(res$data$id, c(1L, 2L, 4L)))
stopifnot(identical(res$data$amount, c(10, NA, NA)))
stopifnot(identical(res$failed$amount, 4L), identical(res$failed$id, integer()))
stopifnot(identical(res$changed, c(id = 2L, amount = 3L, score = 1L)))
stopifnot(nrow(clean.frame(raw, drop.empty = FALSE)$data) == 4L)
stopifnot(inherits(try(clean.frame(raw, types = c(zz = "date")), silent = TRUE), "try-error"))

fc <- coerce.column(c("b", "a", NA, "b"), "factor")
stopifnot(identical(levels(fc), c("b", "a")), identical(attr(fc, "failed"), integer()))
stopifnot(identical(scrub(c(TRUE, NA, FALSE)), c(TRUE, NA, FALSE)))
